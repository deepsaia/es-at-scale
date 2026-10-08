"""Tests for the device-agnostic ES mechanics.

These deliberately use no model and no HuggingFace imports: the ES core is
pure tensor arithmetic and must be verifiable on its own.
"""

import numpy as np
import pytest
import torch

from mps_es.es_loop import apply_update_, perturb_, restore_, zscore


def make_params(device="cpu", dtype=torch.float32):
    """A stand-in for a model's parameter list: mixed shapes, realistic scale."""
    torch.manual_seed(0)
    return [
        (torch.randn(64, 32, dtype=torch.float32) * 0.02).to(device=device, dtype=dtype),
        (torch.randn(32, dtype=torch.float32) * 0.02).to(device=device, dtype=dtype),
        (torch.randn(16, 8, dtype=torch.float32) * 0.02).to(device=device, dtype=dtype),
    ]


def assert_round_trips_within_rounding(params, original, scale):
    """Floating point add/subtract is not invertible, so a perturb/restore round
    trip cannot be bit-exact. What it must be is bounded by the dtype's own
    resolution and utterly negligible against the perturbation scale -- if it is
    not, the noise is not being reconstructed from the seed correctly."""
    for p, o in zip(params, original):
        error = (p - o).abs().max().item()
        tolerance = 4 * torch.finfo(p.dtype).eps * o.abs().max().item()
        assert error <= tolerance, f"{error:.3e} exceeds {tolerance:.3e}"
        assert error < 1e-4 * scale


def test_perturb_then_restore_returns_to_within_rounding_in_fp32():
    params = make_params()
    original = [p.clone() for p in params]

    perturb_(params, seed=1234, scale=0.001)
    restore_(params, seed=1234, scale=0.001)

    assert_round_trips_within_rounding(params, original, scale=0.001)


def test_perturb_actually_changes_the_parameters():
    params = make_params()
    original = [p.clone() for p in params]

    perturb_(params, seed=1234, scale=0.001)

    for p, o in zip(params, original):
        assert not torch.equal(p, o)


def test_same_seed_produces_the_same_perturbation():
    a, b = make_params(), make_params()

    perturb_(a, seed=7, scale=0.001)
    perturb_(b, seed=7, scale=0.001)

    for pa, pb in zip(a, b):
        assert torch.equal(pa, pb)


def test_different_seeds_produce_different_perturbations():
    a, b = make_params(), make_params()

    perturb_(a, seed=7, scale=0.001)
    perturb_(b, seed=8, scale=0.001)

    for pa, pb in zip(a, b):
        assert not torch.equal(pa, pb)


def test_faithful_mode_gives_same_shaped_layers_identical_noise():
    """Upstream re-seeds per layer (worker_extension.py:152), so two layers with
    the same shape receive the same noise. Replicated by default for comparability."""
    params = [torch.zeros(8, 4), torch.zeros(8, 4)]

    perturb_(params, seed=99, scale=1.0, decorrelate_layers=False)

    assert torch.equal(params[0], params[1])


def test_decorrelate_layers_gives_same_shaped_layers_different_noise():
    params = [torch.zeros(8, 4), torch.zeros(8, 4)]

    perturb_(params, seed=99, scale=1.0, decorrelate_layers=True)

    assert not torch.equal(params[0], params[1])


def test_decorrelated_perturb_restore_still_round_trips():
    params = make_params()
    original = [p.clone() for p in params]

    perturb_(params, seed=1234, scale=0.001, decorrelate_layers=True)
    restore_(params, seed=1234, scale=0.001, decorrelate_layers=True)

    assert_round_trips_within_rounding(params, original, scale=0.001)


def test_zscore_has_zero_mean_and_unit_std():
    rewards = [0.1, 0.5, 0.2, 0.9, 0.4]

    z = zscore(rewards)

    assert z.mean() == pytest.approx(0.0, abs=1e-12)
    assert z.std() == pytest.approx(1.0, abs=1e-12)


def test_zscore_returns_zeros_when_all_rewards_are_equal():
    """A degenerate population must not produce NaNs and blow up the weights."""
    z = zscore([0.3, 0.3, 0.3])

    assert np.all(z == 0.0)
    assert not np.any(np.isnan(z))


def test_apply_update_matches_naive_reference():
    """The decomposed seed-by-seed, layer-by-layer update must equal
    theta += alpha * mean_n(z_n * eps_n) computed the obvious way."""
    seeds = [11, 22, 33]
    z = np.array([1.5, -0.5, -1.0])
    alpha = 5e-4

    params = make_params()
    original = [p.clone() for p in params]

    apply_update_(params, seeds=seeds, z_scores=z, alpha=alpha)

    # naive reference: build every noise tensor explicitly and sum
    expected = [o.clone() for o in original]
    for seed, zn in zip(seeds, z):
        for i, e in enumerate(expected):
            g = torch.Generator(device=e.device)
            g.manual_seed(seed)
            noise = torch.randn(e.shape, dtype=e.dtype, device=e.device, generator=g)
            e += (alpha / len(seeds)) * zn * noise

    for p, e in zip(params, expected):
        assert torch.allclose(p, e, atol=1e-9)


def test_apply_update_with_zero_z_scores_leaves_parameters_unchanged():
    params = make_params()
    original = [p.clone() for p in params]

    apply_update_(params, seeds=[1, 2, 3], z_scores=np.zeros(3), alpha=5e-4)

    for p, o in zip(params, original):
        assert torch.allclose(p, o, atol=1e-12)


def test_apply_update_in_bf16_matches_a_float32_accumulated_reference():
    """Upstream sums every seed's share of the update in float32 and casts once.
    Adding each share straight into a bf16 weight would round most of it away,
    because one share is far below the resolution of bf16 at weight scale."""
    seeds = list(range(30))
    z = np.ones(30)  # coherent shares, so the lost signal would be large
    alpha = 0.03
    params = [torch.ones(64, 32, dtype=torch.bfloat16)]
    original = [p.clone() for p in params]

    apply_update_(params, seeds=seeds, z_scores=z, alpha=alpha)

    expected = []
    for o in original:
        total = torch.zeros(o.shape, dtype=torch.float32)
        for seed, zn in zip(seeds, z):
            g = torch.Generator()
            g.manual_seed(seed)
            total += torch.randn(o.shape, dtype=o.dtype, generator=g).float() * zn
        # upstream casts the float32 total to the model dtype, then adds it
        expected.append(o + ((alpha / len(seeds)) * total).to(torch.bfloat16))

    for p, e in zip(params, expected):
        assert torch.equal(p, e)


def test_apply_update_in_bf16_does_not_lose_the_update_to_rounding():
    """Thirty identical shares each below half a bf16 ulp must still add up."""
    seeds = [5] * 30
    params = [torch.ones(64, 32, dtype=torch.bfloat16)]
    original = params[0].clone()

    apply_update_(params, seeds=seeds, z_scores=np.ones(30), alpha=0.03)

    changed = (params[0] != original).float().mean().item()
    assert changed > 0.5, f"only {changed:.0%} of weights moved"
