"""
Stage 3: the training loop.

Ties everything together:
    Grain batches  ->  KimiLinear (GDN-2)  ->  CE + MoE aux loss  ->  AdamW (Optax)
    ->  aux-loss-free router-bias nudge  ->  Orbax checkpoint.

LOSS
    Next-token cross-entropy on the shifted targets, PLUS the summed MoE
    load-balancing aux loss the model already returns. The router-bias update
    (DeepSeek-V3 aux-loss-free balancing) is a NON-gradient step applied after each
    optimizer update, nudging each layer's per-expert selection bias toward uniform
    load using that step's realized `group_sizes`.

OPTIMIZER
    AdamW with a linear warmup + cosine decay schedule and global-norm gradient
    clipping. Weight decay is masked OFF for 1-D params (norms, biases, the GDN-2
    A_log / dt_bias) and applied only to weight matrices — the standard recipe.

MIXED PRECISION
    Governed entirely by model.compute_dtype (fp32 by default; set "bfloat16" on a
    GPU). Master weights stay fp32; logits/loss are fp32 for a stable softmax.

MULTI-GPU (DATA PARALLEL)
    Auto-detected from jax.device_count(): parameters + optimizer state are
    REPLICATED across all visible devices, and each global batch is SHARDED along its
    leading axis (so every GPU processes batch_size/n_devices examples). This is pure
    GSPMD data parallelism — no code path branches on device count; a single device is
    just the degenerate replicate-over-1 case. batch_size must be divisible by the
    number of devices (checked at startup). Used by the 2x-T4 (Kaggle) config.

Run:  python -m pipeline.train --config configs/tiny.yaml [--resume]
"""

from __future__ import annotations

import argparse
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

from kimi_linear_gdn2 import KimiLinear, count_params
from multi_latent_attention.moe import update_router_bias
from pipeline import data as data_mod
from pipeline.checkpointing import CheckpointManager
from pipeline.config import ExperimentConfig


# --------------------------------------------------------------------------- #
#  Model / optimizer construction (shared with evaluate.py).
# --------------------------------------------------------------------------- #
def build_model(cfg: ExperimentConfig, seed: int) -> KimiLinear:
    return KimiLinear(cfg.model, rngs=nnx.Rngs(seed))


def build_schedule(tc) -> optax.Schedule:
    """Linear warmup to `lr`, then cosine decay to `min_lr` over the run."""
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=tc.lr,
        warmup_steps=tc.warmup_steps,
        decay_steps=max(tc.max_steps, tc.warmup_steps + 1),
        end_value=tc.min_lr,
    )


def _weight_decay_mask(params):
    """True on parameters that should be decayed: only >=2-D weight matrices. Norm
    scales, biases and the 1-D GDN-2 decay params (A_log is 2-D but is a gate, not a
    weight matrix — still, treating >=2D uniformly is the conventional choice) are
    left out. Operates on the raw-array pytree Optax sees."""
    return jax.tree.map(lambda p: jnp.ndim(p) >= 2, params)


def build_optimizer(model: KimiLinear, cfg: ExperimentConfig) -> nnx.Optimizer:
    tc = cfg.train
    tx = optax.chain(
        optax.clip_by_global_norm(tc.grad_clip),
        optax.adamw(
            learning_rate=build_schedule(tc),
            b1=tc.beta1,
            b2=tc.beta2,
            eps=tc.eps,
            weight_decay=tc.weight_decay,
            mask=_weight_decay_mask,
        ),
    )
    # Differentiate/optimize ONLY nnx.Param leaves; the MoE router_bias (nnx.Variable)
    # is deliberately excluded — it is updated by the aux-loss-free rule instead.
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


# --------------------------------------------------------------------------- #
#  Loss.
# --------------------------------------------------------------------------- #
def loss_fn(model: KimiLinear, batch: dict[str, jax.Array]):
    """Returns (total_loss, (ce_loss, aux_loss, group_sizes)). total = CE + MoE aux."""
    logits, aux = model(batch["input_ids"])  # logits fp32 [B,L,V]
    ce = optax.softmax_cross_entropy_with_integer_labels(
        logits, batch["target_ids"]).mean()
    total = ce + aux["aux_loss"]
    return total, (ce, aux["aux_loss"], aux["group_sizes"])


@nnx.jit
def train_step(model: KimiLinear, optimizer: nnx.Optimizer,
               batch: dict[str, jax.Array], router_bias_lr: float):
    """One optimizer step + the non-gradient router-bias nudge. Mutates model and
    optimizer in place (nnx tracks the mutations under jit)."""
    (total, (ce, aux_loss, group_sizes)), grads = nnx.value_and_grad(
        loss_fn, has_aux=True)(model, batch)
    optimizer.update(model, grads)

    # Aux-loss-free load balancing: nudge each MoE layer's router bias using this
    # step's realized per-expert token counts (group_sizes[i] for layer i).
    for i, layer in enumerate(model.layers):
        moe = layer.channel_mixer
        moe.router_bias.value = update_router_bias(
            moe.router_bias.value, group_sizes[i], router_bias_lr)

    return total, ce, aux_loss


@nnx.jit
def eval_step(model: KimiLinear, batch: dict[str, jax.Array]):
    """CE loss and token count for one val batch (no aux, no grad)."""
    logits, _ = model(batch["input_ids"])
    tok_ce = optax.softmax_cross_entropy_with_integer_labels(
        logits, batch["target_ids"])  # [B,L]
    return tok_ce.sum(), jnp.array(tok_ce.size, jnp.float32)


def evaluate_loss(model: KimiLinear, val_iter, steps: int, shard=None) -> dict[str, float]:
    """Mean CE / perplexity over `steps` val batches. Sets the model to eval mode so
    any train-only behavior is disabled (harmless here; good hygiene). `shard`, if
    given, places each batch on the data-parallel sharding used by the params."""
    model.eval()
    tot_ce, tot_tok = 0.0, 0.0
    for _ in range(steps):
        batch = _to_jax(next(val_iter))
        if shard is not None:
            batch = shard(batch)
        ce_sum, n = eval_step(model, batch)
        tot_ce += float(ce_sum)
        tot_tok += float(n)
    model.train()
    mean_ce = tot_ce / max(tot_tok, 1.0)
    return {"val_loss": mean_ce, "val_ppl": float(jnp.exp(jnp.array(mean_ce)))}


def _to_jax(batch: dict) -> dict[str, jax.Array]:
    return {k: jnp.asarray(v) for k, v in batch.items()}


# --------------------------------------------------------------------------- #
#  Training loop.
# --------------------------------------------------------------------------- #
def train(cfg: ExperimentConfig, resume: bool = False) -> None:
    tc = cfg.train
    print(f"JAX devices: {jax.devices()}")

    # --- data ---
    meta = data_mod.load_meta(cfg.data.data_dir)
    if meta["vocab_size"] != cfg.model.vocab_size:
        raise ValueError(
            f"model.vocab_size ({cfg.model.vocab_size}) != tokenized vocab "
            f"({meta['vocab_size']}). Fix the YAML so they match.")
    train_iter = data_mod.make_loader(
        cfg.data.data_dir, "train", cfg.data.seq_len, tc.batch_size,
        shuffle=True, repeat=True, seed=cfg.data.shuffle_buffer_seed,
        num_workers=cfg.data.num_workers)
    val_iter = data_mod.make_loader(
        cfg.data.data_dir, "val", cfg.data.seq_len, tc.batch_size,
        shuffle=False, repeat=True, seed=0, num_workers=0)

    # --- model + optimizer ---
    model = build_model(cfg, tc.seed)
    optimizer = build_optimizer(model, cfg)
    print(f"Model params: {count_params(model):,}  "
          f"(compute_dtype={cfg.model.compute_dtype}, seq_len={cfg.data.seq_len})")

    # --- data-parallel sharding (works for 1 device too) ---
    devices = jax.devices()
    n_dev = len(devices)
    if tc.batch_size % n_dev != 0:
        raise ValueError(
            f"batch_size ({tc.batch_size}) must be divisible by the number of "
            f"devices ({n_dev}) for data-parallel training.")
    # AUTO axis type => classic GSPMD auto-partitioning, which resolves the sharded
    # embedding gather / MoE scatter for us (Explicit sharding would demand per-op
    # out-sharding annotations inside the model).
    mesh = jax.make_mesh((n_dev,), ("data",), (AxisType.Auto,))
    data_shard = NamedSharding(mesh, P("data"))  # split batch across devices
    repl_shard = NamedSharding(mesh, P())        # replicate params/opt state
    # Replicate the optimizer (params + Adam state) across all devices. `model` is a
    # submodule of `optimizer`, so this replicates its params in place as well.
    nnx.update(optimizer, jax.device_put(nnx.state(optimizer), repl_shard))
    shard_batch = lambda b: jax.device_put(b, data_shard)  # noqa: E731
    if n_dev > 1:
        print(f"Data-parallel over {n_dev} devices "
              f"({tc.batch_size // n_dev} examples/device).")

    # --- checkpoint manager (+ optional resume) ---
    ckpt = CheckpointManager(tc.ckpt_dir, keep=tc.keep_checkpoints)
    start_step = 0
    if resume and ckpt.latest_step() is not None:
        start_step = ckpt.restore(optimizer) + 1
        print(f"Resumed from step {start_step - 1}")

    # --- loop ---
    t0 = time.time()
    tokens_per_step = tc.batch_size * cfg.data.seq_len
    running_ce = 0.0
    for step in range(start_step, tc.max_steps):
        batch = shard_batch(_to_jax(next(train_iter)))
        total, ce, aux_loss = train_step(model, optimizer, batch, tc.router_bias_lr)
        running_ce += float(ce)

        if (step + 1) % tc.log_every == 0:
            dt = time.time() - t0
            tok_s = tokens_per_step * tc.log_every / dt
            mean_ce = running_ce / tc.log_every
            lr = float(build_schedule(tc)(step))
            print(f"step {step + 1:>7}/{tc.max_steps} | loss {mean_ce:6.4f} | "
                  f"ppl {jnp.exp(jnp.array(mean_ce)):8.2f} | aux {float(aux_loss):.4f} "
                  f"| lr {lr:.2e} | {tok_s:,.0f} tok/s", flush=True)
            running_ce, t0 = 0.0, time.time()

        if (step + 1) % tc.eval_every == 0:
            m = evaluate_loss(model, val_iter, tc.eval_steps, shard=shard_batch)
            print(f"  [eval] step {step + 1} | val_loss {m['val_loss']:.4f} | "
                  f"val_ppl {m['val_ppl']:.2f}", flush=True)
            t0 = time.time()  # don't count eval time against tok/s

        if (step + 1) % tc.save_every == 0:
            ckpt.save(step, optimizer)
            print(f"  [ckpt] saved step {step}", flush=True)

    # final checkpoint
    ckpt.save(tc.max_steps - 1, optimizer)
    ckpt.wait_until_finished()
    print(f"Training complete. Final checkpoint at step {tc.max_steps - 1}.")
    ckpt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Kimi-Linear-GDN2 on CodeParrot.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="Resume from the latest checkpoint in train.ckpt_dir.")
    args = ap.parse_args()
    train(ExperimentConfig.load(args.config), resume=args.resume)


if __name__ == "__main__":
    main()
