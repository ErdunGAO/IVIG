import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit

from src.library.datasets import ihdp, synthetic, tripadvisor
from src.library.datasets.propensity import marginal_logistic_probability


def integral(mean, scale, lower=0.0, upper=1.0):
    return quad(
        lambda u: np.clip(expit(mean + scale * u), lower, upper)
        * np.exp(-u * u / 2) / np.sqrt(2 * np.pi),
        -12, 12, epsabs=1e-11, points=[-6, -3, 0, 3, 6], limit=200,
    )[0]


@pytest.mark.parametrize("scale,lower,upper", [(1.0, 0, 1), (0.55, 0, 1), (0.09, 0.04, 0.88)])
def test_marginal_probability_against_adaptive_integral(scale, lower, upper):
    means = np.linspace(-5, 5, 61)
    expected = [integral(m, scale, lower, upper) for m in means]
    np.testing.assert_allclose(
        marginal_logistic_probability(means, scale, lower, upper), expected, atol=1e-8, rtol=0,
    )


@pytest.mark.parametrize("scenario", synthetic.BASE_SCENARIOS)
def test_synthetic_oracle_integrates_latent_confounder(scenario):
    x, z0, z1, _ = synthetic.generate_binary_pool(scenario, n_pool=4, n_test=3, seed=7)
    # Independent integration through the actual conditional compliance model.
    for z, outcomes in enumerate((z0, z1)):
        for i in range(len(x)):
            value = quad(
                lambda u: float(expit(synthetic.compliance_logit(
                    scenario, np.array([[z]]), np.array([[u]]), x[i:i+1],
                )[0, 0])) * np.exp(-u * u / 2) / np.sqrt(2 * np.pi),
                -10, 10, epsabs=1e-7,
            )[0]
            assert outcomes["P"][i, 0] == pytest.approx(value, abs=1e-7)


def test_tripadvisor_oracle_is_invariant_to_latent_draw():
    x = tripadvisor._sample_source_covariates(8, np.random.RandomState(12))
    first = tripadvisor._generate_pool_outcomes(x, np.random.RandomState(1))
    second = tripadvisor._generate_pool_outcomes(x, np.random.RandomState(2))
    means = tripadvisor._compliance_logits(x, np.zeros(len(x)))
    for z in (0, 1):
        np.testing.assert_array_equal(first[z]["P"], second[z]["P"])
        expected = [integral(m, 0.09, 0.04, 0.88) for m in means[z]]
        np.testing.assert_allclose(first[z]["P"].ravel(), expected, atol=5e-8, rtol=0)
    assert not np.array_equal(first[0]["Y"], second[0]["Y"])


def test_ihdp_truth_and_feedback_are_fixed_per_row(ihdp_csv_factory):
    path = ihdp_csv_factory()
    base = ihdp._load_ihdp_base(path)
    full = ihdp._generate_structural_potentials_for_indices(np.arange(32), path)
    for indices in (np.array([17]), np.array([7, 1, 7, 19]), np.arange(32)[::-1]):
        subset = ihdp._generate_structural_potentials_for_indices(indices, path)
        for z in (0, 1):
            for field in ("TAU", "TAU_BASE", "T", "Y", "P", "U"):
                np.testing.assert_array_equal(subset[z][field], full[z][field][indices])
    for seed, n_pool, n_test in ((0, 32, 8), (1, 12, 40), (3, 16, 16)):
        x, z0, _, test = ihdp.generate_binary_pool(
            ihdp.SCENARIO_NAME, n_pool=n_pool, n_test=n_test, seed=seed, csv_path=path,
        )
        lookup = {tuple(row): tau for row, tau in zip(base["x"], base["tau"])}
        np.testing.assert_array_equal(z0["TAU"].ravel(), [lookup[tuple(row)] for row in x])
        np.testing.assert_array_equal(test.structural.ravel(), [lookup[tuple(row)] for row in test.covariate])


def test_ihdp_oracle_integrates_latent_confounder(ihdp_csv_factory):
    path = ihdp_csv_factory()
    base = ihdp._load_ihdp_base(path)
    score, band, _ = ihdp._band_components(base["x"])
    outcomes = ihdp._generate_structural_potentials_for_indices(np.arange(32), path)
    means = (ihdp._P0_INTERCEPT + ihdp._SCORE_X_WEIGHT * score,
             ihdp._P1_INTERCEPT + ihdp._SCORE_X_WEIGHT * score + ihdp._Z1_BAND_BONUS * band)
    for z in (0, 1):
        expected = [integral(m, ihdp._U_STAGE1_WEIGHT) for m in means[z]]
        np.testing.assert_allclose(outcomes[z]["P"].ravel(), expected, atol=5e-8, rtol=0)


def test_ihdp_cache_tracks_resolved_path(ihdp_csv_factory, monkeypatch):
    first_path = ihdp_csv_factory("first")
    second_path = ihdp_csv_factory("second", effect_offset=5)
    monkeypatch.setenv("IVIG_IHDP_CSV", str(first_path))
    first = ihdp._load_ihdp_base()
    monkeypatch.setenv("IVIG_IHDP_CSV", str(second_path))
    second = ihdp._load_ihdp_base()
    np.testing.assert_allclose(second["tau"] - first["tau"], 5, atol=2e-6)
    assert ihdp._load_ihdp_base(first_path) is first


def test_ihdp_rejects_malformed_csv(tmp_path):
    path = tmp_path / "invalid.csv"
    np.savetxt(path, np.zeros((3, 29)), delimiter=",")
    with pytest.raises(ValueError, match="30 columns"):
        ihdp._load_ihdp_base(path)
