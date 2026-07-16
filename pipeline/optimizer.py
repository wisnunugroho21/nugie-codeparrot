"""
MuonClip optimizer setup for the Kimi-Linear-GDN2 code LM (Flax NNX).

Kimi Linear / Kimi K2 train with Muon and MuonClip; both are implemented FROM
SCRATCH in pipeline/muon.py (Newton-Schulz orthogonalization, the Moonlight
weight-decay + consistent-RMS adjustments, a from-scratch AdamW side, and the
QK-Clip math). This module is the model-aware glue:

  * `_label` decides, per parameter, which side updates it — the Moonlight
    split:

      Muon : parameters that ACT as matrices in a matmul — all Linear kernels,
             and the MoE's stacked expert tensors [E, d_in, d_out], which the
             Newton-Schulz step treats as E independent matrices (it operates
             on the last two axes; leading axes are batch).
      AdamW: everything else — the embedding and LM head (Moonlight keeps both
             on AdamW), and all non-matrix parameters: biases, RMSNorm gains,
             the GDN-2 decay parameters A_log / dt_bias, and the depthwise
             short-conv kernels (shape [width, 1, C] — not a matmul matrix).

  * `make_optimizer` assembles global-norm clip -> muon(...) into an
    nnx.Optimizer. Muon's consistent-RMS scaling matches its update RMS to
    AdamW's ~0.2, so the SAME learning rate / weight decay drive both sides —
    the YAML knobs carry over unchanged. Weight decay touches ONLY the
    Muon-side matrices (embed/head/biases/norms/decays are not pulled to 0).

  * `apply_qk_clip` is the "Clip" in MuonClip (Kimi K2 Sec. 2.1): called by
    the train step right after each optimizer update, it rescales any MLA
    attention head whose max logit exceeded tau in that step's forward pass,
    capping the logits at the weight level. See pipeline/muon.py for why the
    full factor goes on the query projection in our absorbed-MLA form.

Used by pipeline/train.py's build_optimizer; the learning rate / weight decay /
grad-clip / Adam betas / Muon knobs all flow through from TrainConfig.
"""

from __future__ import annotations

import jax
import optax
from flax import nnx

from pipeline.muon import clip_query_kernel, muon, qk_clip_factors


def _path_names(path) -> set[str]:
    """The attribute/key names along one pytree path, e.g. {'layers', '0',
    'token_mixer', 'q_proj', 'kernel', 'value'}."""
    names = set()
    for k in path:
        for attr in ("key", "name", "idx"):
            if hasattr(k, attr):
                names.add(str(getattr(k, attr)))
                break
    return names


def _label(path, leaf) -> str:
    """Classify one parameter: "muon" (a hidden weight matrix) or "adamw"."""
    names = _path_names(path)

    # Moonlight: embedding + LM head stay on AdamW. A_log is 2D [H, dk] but is a
    # per-channel decay parameter, not a matmul weight — AdamW as well.
    if names & {"embed", "lm_head", "A_log"}:
        return "adamw"

    if names & {"w_in", "w_out"} and leaf.ndim == 3:
        return "muon"  # MoE experts: E stacked matrices, batched Newton-Schulz

    if leaf.ndim == 2:
        return "muon"  # every Linear kernel (projections, gates, router, ...)

    # Biases, RMSNorm gains, dt_bias (1D) and depthwise conv kernels (3D but
    # not a matmul matrix) -> AdamW.
    return "adamw"


def _label_tree(params):
    return jax.tree_util.tree_map_with_path(_label, params)


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
    verbose: bool = True,
) -> nnx.Optimizer:
    """Global-norm clip -> Muon (matrices) / AdamW (the rest), NNX-wrapped.

    `learning_rate` may be a float or an Optax schedule; thanks to Muon's
    consistent-RMS scaling it is the SAME learning-rate scale you would give
    AdamW. The other knobs come straight from the run's TrainConfig so the
    YAML still drives the optimizer. wrt=nnx.Param: only Param leaves get
    optimizer state (the MoE router_bias is a plain Variable updated by hand
    in the training loop, so it is correctly left out).
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
            adam_weight_decay=0.0,      # embed/head/biases/norms/decays: no decay
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
            f"AdamW on {n_adam:,} others (embed/head/biases/norms/decays)"
        )

    # Differentiate/optimize ONLY nnx.Param leaves; the MoE router_bias
    # (nnx.Variable) is deliberately excluded — it is updated by the
    # aux-loss-free rule in the training loop instead.
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


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

    Mutates the model in place (like the router-bias nudge in train.py); call
    it right AFTER optimizer.update so the clip sees the just-updated weights'
    logits at the next step.
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
