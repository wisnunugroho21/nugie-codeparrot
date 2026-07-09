# Kimi-Linear (GDN-2) — Code LM training & evaluation pipeline

A from-scratch decoder-only LLM for **program code generation**, built on the
**Kimi Linear** hybrid architecture with **Gated DeltaNet-2** as the linear-attention
token mixer. This repo contains the model *and* a full training/evaluation pipeline
on **CodeParrot**, using **Grain** (data loading), **Optax** (optimization), and
**Orbax** (checkpointing), all in JAX / Flax NNX.

## Architecture (recap)

`kimi_linear_gdn2.py` stacks pre-norm decoder blocks with a 3:1 hybrid attention
schedule:

- **3 of every 4 layers** — Gated DeltaNet-2 linear attention (`gated_deltanet_2/`),
  O(L) with a fixed-size recurrent state.
- **1 of every 4 layers** — NoPE Multi-head Latent Attention (`multi_latent_attention/`).
- **Every layer** — a DeepSeek-V3-style grouped-GEMM MoE channel mixer with
  aux-loss-free load balancing.

## Pipeline layout

```
configs/
  tiny.yaml        # offline smoke test: synthetic data + byte vocab, runs on CPU
  base.yaml        # real run: CodeParrot corpus + BPE tokenizer, bf16, GPU-scale
  colab_t4.yaml    # 1x T4 (Colab), ~190M params, bf16
  kaggle_2xt4.yaml # 2x T4 (Kaggle), data-parallel over both GPUs
  h200.yaml        # 1x H200, ~2.4B-param sparse-MoE run
notebooks/
  colab_train_t4.ipynb     # run the whole pipeline on Google Colab (1x T4)
  kaggle_train_2xt4.ipynb  # run it on Kaggle (2x T4, data-parallel)
pipeline/
  config.py        # typed YAML config (model + data + train), with validation
  tokenizer.py     # byte-level or pretrained codeparrot BPE tokenizer
  prepare_data.py  # stage 1: tokenize CodeParrot -> packed memmap (.bin) + meta.json
  data.py          # stage 2: Grain random-access source + shuffled/batched loader
  train.py         # stage 3: Optax Muon/AdamW loop + router-bias update + Orbax checkpoints
  optimizer.py     # Muon (matrices) / AdamW (rest) split — the Moonlight recipe
  evaluate.py      # stage 4: restore checkpoint -> val loss/ppl + code generation
  checkpointing.py # Orbax CheckpointManager glue for the nnx optimizer state
```

## Install

```bash
pip install -r requirements.txt          # CPU
# GPU: also install a CUDA jax, e.g.  pip install -U "jax[cuda12]"
```

## Quick start (offline, ~minutes on a CPU)

The `tiny` config uses **synthetic random tokens** — no network, no downloads — so
you can verify the whole path works. (Loss stays at ln(256) ≈ 5.55 because random
data has nothing to learn; that is the *correct* result and confirms the mechanics.)

```bash
python -m pipeline.prepare_data --config configs/tiny.yaml
python -m pipeline.train        --config configs/tiny.yaml
python -m pipeline.evaluate     --config configs/tiny.yaml --eval
python -m pipeline.evaluate     --config configs/tiny.yaml --generate --prompt "def f():"
```

## Real training on CodeParrot

The `base` config streams `codeparrot/codeparrot-train-v2-near-dedup` from the HF Hub and uses the
pretrained `codeparrot/codeparrot` BPE tokenizer (vocab 32768).

```bash
python -m pipeline.prepare_data --config configs/base.yaml   # tokenize -> memmap
python -m pipeline.train        --config configs/base.yaml   # train (use a GPU)
python -m pipeline.train        --config configs/base.yaml --resume   # continue

python -m pipeline.evaluate --config configs/base.yaml --eval
python -m pipeline.evaluate --config configs/base.yaml --generate \
    --prompt "def quicksort(arr):" --max-new-tokens 128
```

Raise `data.num_train_docs`, `model.d_model`, `model.n_layers`, and `model.moe_n_routed`
toward the paper's scale as your hardware allows.

## Device configs

| Config | Hardware | Precision | Model | Notes |
|--------|----------|-----------|-------|-------|
| `colab_t4.yaml`    | 1x T4 (16 GB) | bf16 | ~190M, seq 512, batch 8 | T4 has no bf16 tensor cores — bf16 saves memory, not speed. Use `float32` for max stability. |
| `kaggle_2xt4.yaml` | 2x T4 (2x16 GB) | bf16 | same as Colab, batch 16 | **Data-parallel** across both GPUs (8/GPU). Second GPU adds throughput, not model size. |
| `h200.yaml`        | 1x H200 (141 GB) | bf16 | ~2.4B MoE, seq 1024, batch 32 | Native bf16 tensor cores. Scale `seq_len`/experts up; watch O(L²) MLA memory. |

**Notebooks.** `notebooks/colab_train_t4.ipynb` and `notebooks/kaggle_train_2xt4.ipynb`
run the full install → prepare → train → evaluate → generate flow on Colab and Kaggle
respectively (edit the `REPO_URL` cell to point at your clone). The Kaggle one needs
*Accelerator = GPU T4 x2* and *Internet = On*.

**Multi-GPU is automatic.** `train.py` detects `jax.device_count()`, replicates the
params + Adam state on every device, and shards each batch across them (GSPMD data
parallelism). No flag needed — a single GPU is just the replicate-over-1 case.
`batch_size` must be divisible by the device count (validated at startup). The 2x-T4
run prints `Data-parallel over 2 devices (8 examples/device).`

```bash
# On Kaggle's 2x T4, this uses BOTH GPUs with no extra flags:
python -m pipeline.prepare_data --config configs/kaggle_2xt4.yaml
python -m pipeline.train        --config configs/kaggle_2xt4.yaml
```

## How the pieces fit

**Data (Grain).** `prepare_data.py` concatenates every tokenized document (separated
by an EOS id) into one long stream on disk. `data.py` memory-maps it and exposes a
`RandomAccessDataSource` that slices contiguous `(seq_len+1)`-token windows into
`(input_ids, target_ids)` next-token pairs — Grain then shuffles globally, batches,
repeats, and prefetches. `seq_len` must be a multiple of `gdn_chunk_size` (64) and
`≤ max_seq_len`; the config validates this at startup.

**Optimization (Optax).** The Moonlight recipe (`optimizer.py`): **Muon** on the hidden
weight matrices (all Linear kernels + the stacked MoE expert tensors), **AdamW** on
everything else (embedding, LM head, RMSNorm gains, biases, and the GDN-2 decay params).
Muon's consistent-RMS scaling means the AdamW learning rate carries over unchanged. A
linear-warmup → cosine-decay LR schedule and global-norm gradient clipping wrap both;
weight decay applies only to the Muon-side matrices. The loss is next-token cross-entropy
**plus** the MoE load-balancing aux loss the model returns. After each step, the
DeepSeek-V3 **aux-loss-free** router bias is nudged per layer from the realized expert
token counts — a non-gradient update.

**Checkpointing (Orbax).** The entire `nnx.Optimizer` (model params incl. the MoE
router bias, Optax state, and step) is split into its array state and saved by an
Orbax `CheckpointManager` with automatic pruning and `--resume` support.

**Mixed precision.** Controlled by `model.compute_dtype` (`float32` default; set
`bfloat16` on GPU). Master weights stay fp32; the GDN-2 core, RMSNorm, router softmax,
logits, and loss stay fp32 for numerical stability.

## Notes

- Generation uses the model's streaming path: GDN-2 layers carry a fixed-size
  recurrent state (O(1) per token), MLA layers a growing latent cache.
- `model.vocab_size` **must** equal the tokenizer's vocab; `train.py` checks this
  against `meta.json` and aborts on mismatch.
