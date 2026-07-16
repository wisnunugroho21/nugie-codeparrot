"""
MuonClip optimizer setup for the Kimi-Linear-GDN2 code LM (Flax NNX).

Kimi Linear / Kimi K2 train with Muon and MuonClip; both are implemented FROM
SCRATCH in pipeline/muon.py (Newton-Schulz orthogonalization, the Moonlight
weight-decay + consistent-RMS adjustments, a from-scratch AdamW side, and the
QK-Clip math). This module is the model-aware glue:

  * `_label` decides, per parameter, which side updates it — STRICTLY BY
    DIMENSION (a deliberate project choice):

      Muon : every 2D parameter — all Linear kernels (projections, gates,
             router, LM head) and the embedding table.
      AdamW: everything else — 1D params (biases, RMSNorm gains, the GDN-2
             decay params A_log [H] / dt_bias) and 3D params (the stacked MoE
             expert tensors [E, d_in, d_out] and the depthwise short-conv
             kernels [width, 1, C]).

    Note this deviates from the Moonlight/Kimi recipe in two places: they keep
    the (2D) embedding + LM head on AdamW, and they run Muon on the 3D expert
    stacks as E independent matrices (pipeline/muon.py's `orthogonalize` still
    supports that via batching, if the split is ever revisited).

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


def _label(path, leaf) -> str:
    """Classify one parameter strictly by dimension: every 2D param is "muon",
    everything else (1D biases/norms/decays, 3D expert stacks and depthwise
    conv kernels) is "adamw". See the module docstring for the trade-offs."""
    del path  # the rule is purely shape-based
    return "muon" if leaf.ndim == 2 else "adamw"


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
            weight_decay=weight_decay,  # decays ONLY the Muon side (all 2D params)
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
            f"optimizer: Muon on {n_muon:,} 2D params, "
            f"AdamW on {n_adam:,} others (1D biases/norms/decays, 3D experts/conv)"
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
