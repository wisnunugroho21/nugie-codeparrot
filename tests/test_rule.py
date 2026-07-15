"""Verification of the Gated Delta Rule-2 cores (gated_deltanet_2/core.py).

Three layers of evidence:
  1. A float64 numpy oracle — an independent, token-by-token implementation of
     Eq. 9/29 — checks _recurrent_single itself.
  2. Every chunkwise core is checked against the recurrent core on the same fp32
     inputs, under moderate decay (all cores in range) and non-zero S0.
  3. The overflow-safe cores (centered / pairwise / subchunking) are additionally
     checked under strong decay, where the faithful factors exp(-G) overflow fp32.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from gated_deltanet_2.core import (
    _CHUNKWISE_CORES,
    _recurrent_single,
    chunkwise_gated_delta_rule_2,
    recurrent_gated_delta_rule_2,
)

ALL_CORES = sorted(_CHUNKWISE_CORES)
SAFE_CORES = ["centered", "pairwise", "subchunking"]  # no |G_C| ~ 88 fp32 limit


# --------------------------------------------------------------------------- #
#  Input generation.
# --------------------------------------------------------------------------- #
def make_inputs(seed, L=64, dk=8, dv=8, g_lo=-0.4, g_hi=-0.01, s0_scale=1.0):
    """Random single-head inputs with realistic magnitudes: g <= 0 log-decay,
    b, w in [0, 1] (sigmoid range), unit-ish q/k/v. Returns numpy float32."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(L, dk)).astype(np.float32)
    k = rng.normal(size=(L, dk)).astype(np.float32)
    v = rng.normal(size=(L, dv)).astype(np.float32)
    g = rng.uniform(g_lo, g_hi, size=(L, dk)).astype(np.float32)
    b = rng.uniform(0.0, 1.0, size=(L, dk)).astype(np.float32)
    w = rng.uniform(0.0, 1.0, size=(L, dv)).astype(np.float32)
    S0 = (s0_scale * rng.normal(size=(dk, dv))).astype(np.float32)
    return q, k, v, g, b, w, S0


def oracle_f64(q, k, v, g, b, w, S0):
    """Independent float64 numpy scan of Eq. 9/29 — no JAX, no shared code."""
    q, k, v, g, b, w = (np.asarray(x, np.float64) for x in (q, k, v, g, b, w))
    S = np.asarray(S0, np.float64).copy()
    L, dv = v.shape
    out = np.empty((L, dv), np.float64)
    for t in range(L):
        S = np.exp(g[t])[:, None] * S              # forget: Diag(alpha) S
        r = S.T @ (b[t] * k[t])                    # recall along e = b*k
        S = S + np.outer(k[t], w[t] * v[t] - r)    # delta write of the residual
        out[t] = S.T @ q[t]                        # read with q
    return out, S


# --------------------------------------------------------------------------- #
#  1. The recurrent core against the independent float64 oracle.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1])
def test_recurrent_matches_f64_oracle(seed):
    inputs = make_inputs(seed)
    o_ref, S_ref = oracle_f64(*inputs)
    o, S = _recurrent_single(*(jnp.asarray(x) for x in inputs))
    # rtol 5e-4: the fp32 scan accumulates rounding over L=64 sequential
    # delta-rule updates against the float64 reference.
    np.testing.assert_allclose(np.asarray(o), o_ref, rtol=5e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(S), S_ref, rtol=5e-4, atol=1e-4)


# --------------------------------------------------------------------------- #
#  2. Every chunkwise core against the recurrent core (moderate decay).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("core", ALL_CORES)
@pytest.mark.parametrize("seed", [0, 1])
def test_chunkwise_matches_recurrent(core, seed):
    q, k, v, g, b, w, S0 = (jnp.asarray(x) for x in make_inputs(seed))
    # Add [B=1, H=1] axes for the public entry points.
    args = tuple(x[None, None] for x in (q, k, v, g, b, w, S0))
    o_ref, S_ref = recurrent_gated_delta_rule_2(*args)
    o, S = chunkwise_gated_delta_rule_2(
        *args, chunk_size=16, core=core, sub_chunk_size=4)
    np.testing.assert_allclose(np.asarray(o), np.asarray(o_ref),
                               rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(np.asarray(S), np.asarray(S_ref),
                               rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("core", ALL_CORES)
def test_state_carry_across_calls(core):
    """Running two half-sequences chained through S equals one full run —
    the property the layer's step() prefill relies on."""
    q, k, v, g, b, w, S0 = (jnp.asarray(x)[None, None]
                            for x in make_inputs(7, L=64))
    kw = dict(chunk_size=16, core=core, sub_chunk_size=4)
    o_full, S_full = chunkwise_gated_delta_rule_2(q, k, v, g, b, w, S0, **kw)

    half = 32
    o1, S_mid = chunkwise_gated_delta_rule_2(
        q[:, :, :half], k[:, :, :half], v[:, :, :half],
        g[:, :, :half], b[:, :, :half], w[:, :, :half], S0, **kw)
    o2, S_end = chunkwise_gated_delta_rule_2(
        q[:, :, half:], k[:, :, half:], v[:, :, half:],
        g[:, :, half:], b[:, :, half:], w[:, :, half:], S_mid, **kw)

    np.testing.assert_allclose(
        np.asarray(jnp.concatenate([o1, o2], axis=2)), np.asarray(o_full),
        rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.asarray(S_end), np.asarray(S_full),
                               rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------- #
#  3. Strong decay: the overflow-safe cores where the faithful factors blow up.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("core", SAFE_CORES)
def test_safe_cores_under_strong_decay(core):
    """g in [-8, -4] with C=16 gives per-chunk |G_C| up to ~128 > 88: the
    faithful exp(-G) overflows fp32, the safe cores must still be exact."""
    inputs = make_inputs(3, g_lo=-8.0, g_hi=-4.0)
    args = tuple(jnp.asarray(x)[None, None] for x in inputs)
    o_ref, S_ref = recurrent_gated_delta_rule_2(*args)
    o, S = chunkwise_gated_delta_rule_2(
        *args, chunk_size=16, core=core, sub_chunk_size=4)
    assert bool(jnp.all(jnp.isfinite(o))) and bool(jnp.all(jnp.isfinite(S)))
    np.testing.assert_allclose(np.asarray(o), np.asarray(o_ref),
                               rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(np.asarray(S), np.asarray(S_ref),
                               rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("core", ["pairwise", "subchunking"])
def test_rangeless_cores_under_extreme_decay(core):
    """g in [-16, -12] with C=16: per-chunk |G_C| up to ~256, beyond even the
    centered core's ~176 range. Only the range-free cores must survive."""
    inputs = make_inputs(4, g_lo=-16.0, g_hi=-12.0)
    args = tuple(jnp.asarray(x)[None, None] for x in inputs)
    o_ref, S_ref = recurrent_gated_delta_rule_2(*args)
    o, S = chunkwise_gated_delta_rule_2(
        *args, chunk_size=16, core=core, sub_chunk_size=4)
    assert bool(jnp.all(jnp.isfinite(o))) and bool(jnp.all(jnp.isfinite(S)))
    np.testing.assert_allclose(np.asarray(o), np.asarray(o_ref),
                               rtol=1e-3, atol=1e-4)


# --------------------------------------------------------------------------- #
#  Entry-point plumbing.
# --------------------------------------------------------------------------- #
def test_batched_shapes_and_batch_independence():
    """[B, H] batching is pure vmap plumbing: each (b, h) slice must equal the
    single-head run of its own inputs."""
    B, H, L, dk, dv = 2, 3, 32, 8, 4
    rng = np.random.default_rng(11)
    q, k, g, b = (jnp.asarray(rng.normal(size=(B, H, L, dk)), jnp.float32)
                  for _ in range(4))
    g = -jnp.abs(g) * 0.1
    b = jax.nn.sigmoid(b)
    v = jnp.asarray(rng.normal(size=(B, H, L, dv)), jnp.float32)
    w = jax.nn.sigmoid(jnp.asarray(rng.normal(size=(B, H, L, dv)), jnp.float32))
    S0 = jnp.asarray(rng.normal(size=(B, H, dk, dv)), jnp.float32)

    o, S = chunkwise_gated_delta_rule_2(q, k, v, g, b, w, S0, chunk_size=16)
    assert o.shape == (B, H, L, dv) and S.shape == (B, H, dk, dv)

    o01, S01 = _recurrent_single(q[0, 1], k[0, 1], v[0, 1],
                                 g[0, 1], b[0, 1], w[0, 1], S0[0, 1])
    np.testing.assert_allclose(np.asarray(o[0, 1]), np.asarray(o01),
                               rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(np.asarray(S[0, 1]), np.asarray(S01),
                               rtol=1e-3, atol=1e-4)


def test_validation_errors():
    args = tuple(jnp.asarray(x)[None, None] for x in make_inputs(0, L=64))
    with pytest.raises(ValueError, match="divisor"):
        chunkwise_gated_delta_rule_2(*args, chunk_size=48)  # 64 % 48 != 0
    with pytest.raises(ValueError, match="core="):
        chunkwise_gated_delta_rule_2(*args, chunk_size=16, core="nope")
    with pytest.raises(ValueError, match="sub_chunk_size"):
        chunkwise_gated_delta_rule_2(
            *args, chunk_size=16, core="subchunking", sub_chunk_size=5)
