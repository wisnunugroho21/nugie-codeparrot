"""
Muon and MuonClip, implemented from scratch (JAX, Optax-compatible).

Kimi Linear / Kimi K2 train with Muon ("Muon is Scalable for LLM Training",
arXiv:2502.16982 — "Moonlight") and its successor MuonClip (the Kimi K2 report,
arXiv:2507.20534). This module reimplements both rather than delegating to
`optax.contrib.muon`, exposing every piece as a small Optax
`GradientTransformation` so the algorithm is visible end to end.

MUON (one matrix parameter W, gradient g)
-----------------------------------------
    M_t = mu * M_{t-1} + g_t                     # SGD-style momentum
    O_t = NewtonSchulz(g_t + mu * M_t)           # orthogonalize (Nesterov form)
    W  <- W - lr * (0.2 * sqrt(max(m, n)) * O_t + wd * W)

Newton-Schulz runs a few iterations of the quintic map
    X <- a X + (b A + c A^2) X,   A = X X^T
which pushes every singular value of the (normalized) momentum toward 1 — an
approximation of U V^T from the SVD of the momentum. The update then moves
every direction with equal strength (steepest descent under the spectral norm)
instead of being dominated by a few large singular values. Orthogonalization
is only defined for matrices, so Muon covers matrix parameters and everything
else falls back to AdamW; WHICH params count as matrices is the caller's
policy, passed to `muon()` as `param_labels` (this project's rule — strictly
2D -> Muon, all else -> AdamW — lives in pipeline/optimizer.py). Both
Moonlight adjustments are included:

  * weight decay inside the Muon update (their Sec. 2.2), and
  * consistent-RMS scaling by 0.2 * sqrt(max(fan_in, fan_out)), which matches
    the update RMS to AdamW's empirical ~0.2 for any matrix shape — so Muon
    reuses AdamW's learning rate and weight decay values unchanged.

`orthogonalize` operates on the LAST TWO axes and treats leading axes as a
batch, so a stacked MoE expert tensor [E, d_in, d_out] is handled as E
independent matrices with no special casing.

MUONCLIP = MUON + QK-CLIP (Kimi K2, Sec. 2.1)
---------------------------------------------
Muon-trained attention is prone to exploding attention logits. QK-Clip caps
them at the WEIGHT level: after each optimizer step, any attention head whose
max logit S_h (observed in that step's forward pass) exceeded a threshold tau
gets its query/key projections rescaled by gamma_h = tau / S_h, so the head's
logits are pulled back to at most tau while its attention pattern's shape is
preserved. Heads that stayed under tau are untouched, and once training
stabilizes (S_h <= tau everywhere) QK-Clip becomes a no-op.

This file holds the model-agnostic math (`qk_clip_factors`, per-head kernel
rescaling); the model-aware pieces live next door:
  * multi_latent_attention/attention.py surfaces the per-head max logits,
  * pipeline/optimizer.py's `apply_qk_clip` walks the model's MLA layers,
  * pipeline/train.py calls it right after each `optimizer.update`.

ONE DEVIATION, FORCED BY THE ABSORBED MLA: Kimi K2 splits gamma_h between the
query and key projections (sqrt(gamma) each). Our MLA layers are written in
the absorbed NoPE form where the KV latent serves as BOTH keys and values
(see attention.py) — scaling the latent projection would corrupt the values.
Since a QK logit is bilinear in (W_q, W_k), applying the FULL factor gamma_h
to the query projection alone caps the logits identically without touching
the value path. The q/k split in the paper only balances the two matrices'
norms; the clipping effect is the same.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple, Union

import jax
import jax.numpy as jnp
import optax

ScalarOrSchedule = Union[float, optax.Schedule]

# Quintic Newton-Schulz coefficients (Keller Jordan's Muon; also used by
# Moonlight and optax.contrib). Tuned to maximize convergence speed of the
# small singular values at x=0 rather than to converge tightly to 1: after 5
# steps every singular value lands in roughly (0.7, 1.2), which is all the
# update direction needs.
_NS_COEFFS = (3.4445, -4.7750, 2.0315)


# --------------------------------------------------------------------------- #
#  Newton-Schulz orthogonalization.
# --------------------------------------------------------------------------- #
def orthogonalize(g: jax.Array, ns_steps: int = 5, eps: float = 1e-7) -> jax.Array:
    """Approximate U V^T of the SVD of `g` via Newton-Schulz iteration.

    Acts on the last two axes; leading axes are batch (matmul broadcasting),
    so [m, n] and stacked [E, m, n] both work. Wide orientation (rows <= cols)
    keeps the Gram matrix A = X X^T at the smaller of the two square sizes;
    tall inputs are transposed in and back out. Runs in fp32 regardless of the
    input dtype (5 chained matmul steps amplify bf16 rounding).
    """
    if g.ndim < 2:
        raise ValueError(f"orthogonalize needs a matrix, got shape {g.shape}")
    a, b, c = _NS_COEFFS

    x = g.astype(jnp.float32)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.swapaxes(-2, -1)

    # Normalize so every singular value is <= 1 (Frobenius >= spectral norm):
    # the NS iteration only converges toward 1 from inside [0, sqrt(3)].
    x = x / (jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) + eps)

    def body(x, _):
        gram = x @ x.swapaxes(-2, -1)  # A = X X^T, per batch element
        return a * x + (b * gram + c * (gram @ gram)) @ x, None

    x, _ = jax.lax.scan(body, x, None, length=ns_steps)

    if transposed:
        x = x.swapaxes(-2, -1)
    return x


# --------------------------------------------------------------------------- #
#  The Muon side: momentum -> Newton-Schulz -> consistent-RMS scale.
# --------------------------------------------------------------------------- #
class MuonState(NamedTuple):
    momentum: Any  # same pytree structure as the params


def scale_by_muon(
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
    consistent_rms: float = 0.2,
) -> optax.GradientTransformation:
    """Muon's direction: orthogonalized momentum, RMS-matched to AdamW.

    Momentum is the SGD-style accumulator M <- beta*M + g (not an EMA); with
    `nesterov` the orthogonalized input is g + beta*M. The output is scaled by
    consistent_rms * sqrt(max(fan_in, fan_out)) so its RMS is ~consistent_rms
    for any matrix shape (Moonlight Sec. 2.2) — an orthogonal(ish) [m, n]
    matrix has RMS ~ 1/sqrt(max(m, n)).
    """

    def init_fn(params):
        return MuonState(momentum=jax.tree.map(jnp.zeros_like, params))

    def update_fn(updates, state, params=None):
        del params
        momentum = jax.tree.map(
            lambda m, g: beta * m + g, state.momentum, updates
        )

        def one(g, m):
            u = g + beta * m if nesterov else m
            o = orthogonalize(u, ns_steps)
            scale = consistent_rms * math.sqrt(max(u.shape[-2], u.shape[-1]))
            return (scale * o).astype(g.dtype)

        return jax.tree.map(one, updates, momentum), MuonState(momentum)

    return optax.GradientTransformation(init_fn, update_fn)


# --------------------------------------------------------------------------- #
#  The AdamW side (for the non-matrix params), also from scratch.
# --------------------------------------------------------------------------- #
class AdamState(NamedTuple):
    count: jax.Array  # int32 scalar: number of steps taken
    mu: Any  # first-moment EMA
    nu: Any  # second-moment EMA


def scale_by_adam(
    b1: float = 0.9, b2: float = 0.95, eps: float = 1e-8
) -> optax.GradientTransformation:
    """Adam's direction: bias-corrected mu / (sqrt(nu) + eps)."""

    def init_fn(params):
        return AdamState(
            count=jnp.zeros([], jnp.int32),
            mu=jax.tree.map(jnp.zeros_like, params),
            nu=jax.tree.map(jnp.zeros_like, params),
        )

    def update_fn(updates, state, params=None):
        del params
        count = state.count + 1
        mu = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, state.mu, updates)
        nu = jax.tree.map(
            lambda v, g: b2 * v + (1 - b2) * jnp.square(g), state.nu, updates
        )
        c = count.astype(jnp.float32)
        bc1, bc2 = 1 - b1**c, 1 - b2**c  # bias corrections
        updates = jax.tree.map(
            lambda m, v: (m / bc1) / (jnp.sqrt(v / bc2) + eps), mu, nu
        )
        return updates, AdamState(count, mu, nu)

    return optax.GradientTransformation(init_fn, update_fn)


# --------------------------------------------------------------------------- #
#  Shared tail pieces: decoupled weight decay + (scheduled) learning rate.
# --------------------------------------------------------------------------- #
def add_weight_decay(weight_decay: float) -> optax.GradientTransformation:
    """Decoupled (AdamW-style) weight decay: update <- update + wd * param."""

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError("add_weight_decay requires params.")
        updates = jax.tree.map(lambda u, p: u + weight_decay * p, updates, params)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


class ScaleByLrState(NamedTuple):
    count: jax.Array  # int32 scalar: steps taken (drives the schedule)


def scale_by_lr(learning_rate: ScalarOrSchedule) -> optax.GradientTransformation:
    """update <- -lr * update, where lr may be a constant or an Optax schedule."""

    def init_fn(params):
        del params
        return ScaleByLrState(count=jnp.zeros([], jnp.int32))

    def update_fn(updates, state, params=None):
        del params
        lr = learning_rate(state.count) if callable(learning_rate) else learning_rate
        updates = jax.tree.map(lambda u: -lr * u, updates)
        return updates, ScaleByLrState(state.count + 1)

    return optax.GradientTransformation(init_fn, update_fn)


# --------------------------------------------------------------------------- #
#  The full optimizer: Muon on matrix params, AdamW on the rest.
# --------------------------------------------------------------------------- #
def muon(
    learning_rate: ScalarOrSchedule,
    param_labels,
    *,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
    consistent_rms: float = 0.2,
    weight_decay: float = 0.0,
    adam_b1: float = 0.9,
    adam_b2: float = 0.95,
    eps: float = 1e-8,
    adam_weight_decay: float = 0.0,
) -> optax.GradientTransformation:
    """Our Muon optimizer (drop-in for `optax.contrib.muon`).

    `param_labels` is a pytree of {"muon", "adamw"} matching the params — or a
    callable producing one — deciding which side updates each leaf. Thanks to
    consistent-RMS scaling, one `learning_rate` (constant or schedule) drives
    both sides. `weight_decay` applies to the Muon-side matrices,
    `adam_weight_decay` to the rest.
    """
    muon_side = optax.chain(
        scale_by_muon(beta, ns_steps, nesterov, consistent_rms),
        add_weight_decay(weight_decay),
        scale_by_lr(learning_rate),
    )
    adam_side = optax.chain(
        scale_by_adam(adam_b1, adam_b2, eps),
        add_weight_decay(adam_weight_decay),
        scale_by_lr(learning_rate),
    )
    return optax.multi_transform(
        {"muon": muon_side, "adamw": adam_side}, param_labels
    )


# --------------------------------------------------------------------------- #
#  QK-Clip (the "Clip" in MuonClip) — model-agnostic math.
# --------------------------------------------------------------------------- #
def qk_clip_factors(max_logits: jax.Array, tau: float) -> jax.Array:
    """Per-head rescale factor gamma_h = min(1, tau / S_h), elementwise.

    `max_logits` are the per-head max attention logits S_h observed in the
    step's forward pass (post 1/sqrt(d) scaling — the actual softmax inputs).
    Heads at or under `tau` get exactly 1.0 (untouched); guarded via `where`
    so a non-positive S_h can never produce a negative or exploding factor.
    """
    return jnp.where(max_logits > tau, tau / max_logits, 1.0)


def clip_query_kernel(
    kernel: jax.Array, gammas: jax.Array, head_dim: int
) -> jax.Array:
    """Rescale a fused per-head query projection kernel [d_in, H * head_dim]
    by gamma_h on each head's column block. Logits are linear in this kernel,
    so head h's max logit becomes exactly gamma_h * S_h <= tau."""
    d_in = kernel.shape[0]
    per_head = kernel.reshape(d_in, -1, head_dim) * gammas[None, :, None]
    return per_head.reshape(kernel.shape).astype(kernel.dtype)
