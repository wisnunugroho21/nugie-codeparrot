"""
MuonClip optimizer setup for the Kimi-Linear-GDN2 code LM (Flax NNX).

Kimi Linear / Kimi K2 train with Muon and MuonClip; both are implemented FROM
SCRATCH in pipeline/muon.py (Newton-Schulz orthogonalization, the Moonlight
weight-decay + consistent-RMS adjustments, a from-scratch AdamW side, and the
QK-Clip math). This module is the model-aware glue:

  * `_label` decides, per parameter, which side updates it — by dimension,
    with the Moonlight/Kimi embedding/head exception:

      Muon : every 2D parameter EXCEPT the embedding table and the LM head —
             all Linear kernels (projections, gates, router).
      AdamW: everything else — the embedding + LM head (Moonlight/Kimi keep
             both on AdamW: the embedding's gradient is row-sparse, and
             orthogonalizing it smears a full-RMS update across every vocab
             row, including tokens absent from the batch), 1D params (biases,
             RMSNorm gains, the GDN-2 decay params A_log [H] / dt_bias) and
             3D params (the stacked MoE expert tensors [E, d_in, d_out] and
             the depthwise short-conv kernels [width, 1, C]).

    One remaining deviation from the Moonlight/Kimi recipe: they run Muon on
    the 3D expert stacks as E independent matrices (pipeline/muon.py's
    `orthogonalize` still supports that via batching, if the split is ever
    revisited).

  * `MuonClipOptimizer` packages MuonClip as ONE optimizer, the way Kimi K2
    presents it (Sec. 2.1): a single `update(model, grads, max_logits)` call
    runs the gradient update (global-norm clip -> Muon/AdamW) and then
    QK-Clip — rescaling any MLA attention head whose max logit exceeded tau
    in that step's forward pass, capping the logits at the weight level. The
    optimizer owns tau; pass qk_clip_tau=None (or omit max_logits) for plain
    Muon with no clipping.

  * `make_optimizer` assembles global-norm clip -> muon(...) into a
    MuonClipOptimizer. Muon's consistent-RMS scaling matches its update RMS
    to AdamW's ~0.2, so the SAME learning rate / weight decay drive both
    sides — the YAML knobs carry over unchanged. Weight decay touches ONLY
    the Muon-side matrices (embed/head/biases/norms/decays are not pulled
    to 0).

  * `apply_qk_clip` is the "Clip" half, kept callable on its own: it walks
    the model's MLA layers and rescales each exceeding head. See
    pipeline/muon.py for why the full factor goes on the query projection in
    our absorbed-MLA form.

Used by pipeline/train.py's build_optimizer; the learning rate / weight decay /
grad-clip / Adam betas / Muon knobs all flow through from TrainConfig.
"""

from __future__ import annotations

import jax
import optax
from flax import nnx

from pipeline.muon import clip_query_kernel, muon, qk_clip_factors


# 2D params exempted from Muon (path components), per the Moonlight/Kimi
# recipe: the embedding table and the untied LM head stay on AdamW.
_ADAMW_2D = ("embed", "lm_head")


def _label(path, leaf) -> str:
    """Classify one parameter: every 2D param is "muon" — except the embedding
    and LM head, which follow Moonlight/Kimi onto the AdamW side — and
    everything else (1D biases/norms/decays, 3D expert stacks and depthwise
    conv kernels) is "adamw". See the module docstring for the trade-offs."""
    if leaf.ndim != 2:
        return "adamw"
    names = [str(getattr(k, a)) for k in path
             for a in ("key", "name", "idx") if hasattr(k, a)]
    return "adamw" if any(n in _ADAMW_2D for n in names) else "muon"


def _label_tree(params):
    return jax.tree_util.tree_map_with_path(_label, params)


class MuonClipOptimizer(nnx.Optimizer):
    """MuonClip as a single optimizer (Kimi K2 Sec. 2.1): Muon + QK-Clip.

    `update(model, grads, max_logits)` applies the wrapped gradient
    transformation (global-norm clip -> Muon on matrices / AdamW on the rest)
    and then QK-Clip: any MLA head whose max attention logit `max_logits`
    exceeded `qk_clip_tau` in the step's forward pass has its query
    projection rescaled so the logits stay capped. `max_logits` is the
    model's aux["mla_max_logits"] ([n_mla, Hq]; under data parallelism,
    pmax-ed across replicas first).

    QK-Clip is a WEIGHT-level edit, not a gradient term, so it lives here as
    a post-update mutation rather than inside the optax chain. It degrades
    gracefully: `qk_clip_tau=None` or omitting `max_logits` gives plain Muon.
    `qk_clip_tau` is static Python metadata (not nnx state), so checkpoints
    and replication treat this exactly like a plain nnx.Optimizer.
    """

    def __init__(self, model: nnx.Module, tx, *,
                 qk_clip_tau: float | None = None, wrt=nnx.Param):
        super().__init__(model, tx, wrt=wrt)
        self.qk_clip_tau = qk_clip_tau

    def update(self, model: nnx.Module, grads,
               max_logits: jax.Array | None = None, **kwargs) -> None:
        super().update(model, grads, **kwargs)
        if self.qk_clip_tau is not None and max_logits is not None:
            apply_qk_clip(model, max_logits, self.qk_clip_tau)


def make_optimizer(
    model: nnx.Module,
    learning_rate,
    *,
    weight_decay: float = 0.01,
    clip_norm: float = 1.0,
    adam_b1: float = 0.9,
    adam_b2: float = 0.95,
    eps: float = 1e-8,
    muon_beta: float = 0.95,
    muon_ns_steps: int = 5,
    qk_clip_tau: float | None = None,
    verbose: bool = True,
) -> MuonClipOptimizer:
    """Global-norm clip -> Muon (matrices) / AdamW (the rest) + QK-Clip,
    packaged as a MuonClipOptimizer.

    `learning_rate` may be a float or an Optax schedule; thanks to Muon's
    consistent-RMS scaling it is the SAME learning-rate scale you would give
    AdamW. `qk_clip_tau` arms QK-Clip (None = plain Muon, no clipping). The
    other knobs come straight from the run's TrainConfig so the YAML still
    drives the optimizer. wrt=nnx.Param: only Param leaves get optimizer
    state (the MoE router_bias is a plain Variable updated by hand in the
    training loop, so it is correctly left out).
    """
    tx = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        muon(
            learning_rate,
            _label_tree,
            beta=muon_beta,             # Muon momentum (matrices)
            ns_steps=muon_ns_steps,     # Newton-Schulz iterations
            weight_decay=weight_decay,  # decays ONLY the Muon-side matrices
            adam_b1=adam_b1,
            adam_b2=adam_b2,
            eps=eps,
            adam_weight_decay=0.0,      # AdamW side (1D/3D params): no decay
            consistent_rms=0.2,         # Moonlight: match AdamW's update RMS
        ),
    )

    if verbose:
        params = nnx.state(model, nnx.Param)
        leaves = jax.tree_util.tree_leaves_with_path(params)
        n_muon = sum(l.size for p, l in leaves if _label(p, l) == "muon")
        n_adam = sum(l.size for p, l in leaves if _label(p, l) == "adamw")
        print(
            f"optimizer: Muon on {n_muon:,} matrix params, "
            f"AdamW on {n_adam:,} others (embed/head, 1D biases/norms/decays, "
            f"3D experts/conv)"
        )

    # Differentiate/optimize ONLY nnx.Param leaves; the MoE router_bias
    # (nnx.Variable) is deliberately excluded — it is updated by the
    # aux-loss-free rule in the training loop instead.
    return MuonClipOptimizer(model, tx, qk_clip_tau=qk_clip_tau, wrt=nnx.Param)


# --------------------------------------------------------------------------- #
#  QK-Clip: the post-step weight rescale that makes Muon -> MuonClip.
# --------------------------------------------------------------------------- #
def apply_qk_clip(model: nnx.Module, max_logits: jax.Array, tau: float) -> None:
    """Rescale each MLA head whose max attention logit exceeded `tau`.

    `max_logits` is the model forward's aux["mla_max_logits"]: [n_mla, Hq],
    one row per full-attention layer in layer order (under data parallelism,
    already pmax-ed across replicas). For each layer, head h is rescaled by
    gamma_h = min(1, tau / S_h) on its w_q_uk column block; logits are linear
    in that kernel, so the head's max logit becomes exactly min(S_h, tau).
    Heads under the threshold get gamma = 1 (a no-op) — as training
    stabilizes, this whole function converges to the identity.

    Mutates the model in place (like the router-bias nudge in train.py).
    MuonClipOptimizer.update calls this right AFTER the gradient update so
    the clip sees the just-updated weights' logits at the next step; it is
    kept independently callable for tests and manual use.
    """
    mla_layers = [layer for layer in model.layers if layer.is_full_attn]
    if len(mla_layers) != max_logits.shape[0]:
        raise ValueError(
            f"max_logits has {max_logits.shape[0]} rows but the model has "
            f"{len(mla_layers)} full-attention layers."
        )
    for i, layer in enumerate(mla_layers):
        attn = layer.token_mixer
        gammas = qk_clip_factors(max_logits[i], tau)  # [Hq]
        attn.w_q_uk.kernel.set_value(
            clip_query_kernel(
                attn.w_q_uk.kernel.get_value(), gammas, attn.head_dim
            )
        )
