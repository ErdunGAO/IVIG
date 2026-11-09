import numpy as np
import pytest
import torch
from src.library.models.gpiv import GPIVTrainer
from src.library.datasets import synthetic
from src.library.datasets.active_learning import ActiveIVDataset
from src.library import acquisitions
from src.library.metrics import weighted_sqrt_pehe_np


@pytest.fixture(scope='module')
def fitted():
    np.random.seed(7)
    torch.manual_seed(7)
    x, z0, z1, test = synthetic.generate_binary_pool('sharp', n_pool=16, d_covariate=5, n_test=6, seed=7)
    data = ActiveIVDataset(x)
    idx = np.arange(8)
    z = np.arange(8) % 2
    t = np.where(z[:, None] == 0, z0['T'][idx], z1['T'][idx])
    y = np.where(z[:, None] == 0, z0['Y'][idx], z1['Y'][idx])
    data.acquire(idx, z, t, y)
    trainer = GPIVTrainer({'n_epoch': 2, 'lr': 0.05}, gpu_flg=False)
    trainer.train(7, train_data=data.get_train_data(), test_data=test)
    return trainer, data, test


def test_estimator_returns_effect_rmse(fitted):
    trainer, _, test = fitted
    pred = trainer.mdl.predict_cate(test.covariate)
    expected = weighted_sqrt_pehe_np(pred, test.structural)
    assert trainer.evaluate_cate(test) == pytest.approx(expected)
    assert trainer.evaluate(test) == pytest.approx(expected)


def test_fantasy_covariance_matches_direct_gaussian_conditioning(fitted):
    trainer, data, test = fitted
    m = trainer.mdl
    target = m._ensure_2d_covariate(test.covariate[:2])
    query = m._ensure_2d_covariate(data.get_pool_x()[:1])
    z = m._ensure_2d_instrument(1., 1)
    with torch.no_grad():
        aug = m._candidate_augmented_from_branch(query, z, assumed_t=1)
        a1, a0 = m._build_do_augmented(target)
        c = m._posterior_covariance(torch.cat([a1, a0, aug]))
        contrast = torch.tensor([[1., 0., -1., 0.], [0., 1., 0., -1.]], dtype=m.dtype)
        prior = contrast @ c[:4, :4] @ contrast.T
        cross = contrast @ c[:4, 4:]
        posterior = prior - cross @ torch.linalg.solve(c[4:, 4:] + m.noise_var * torch.eye(1, dtype=m.dtype), cross.T)
        expected = posterior.diag().numpy() * m.y_std**2
        actual = m._fantasy_cate_variance_from_query_aug_batch(target, aug).numpy()[0]
    assert actual == pytest.approx(expected, rel=1e-8, abs=1e-10)


def test_single_particle_recovers_plugin_score(fitted):
    trainer, data, test = fitted
    m = trainer.mdl
    x = data.get_pool_x()[:2]
    a = m.phantom_ivig_scores(x, 1., test.covariate[:2])
    b = m.krr_particle_ivig_scores(x, 1., test.covariate[:2], n_particles=1)
    assert b == pytest.approx(a, rel=1e-8, abs=1e-10)


@pytest.mark.parametrize('name', ['random', 'mu_bald', 'tau_bald', 'mu_rho_bald', 'gamma', 'ivig',
                                'y_ivig', 'no_stage1_ivig', 'krr_mc_ivig'])
def test_acquisitions_return_distinct_binary_queries(fitted, name):
    trainer, data, test = fitted
    np.random.seed(3)
    indices, z = acquisitions.FUNCTIONS[name](trainer, data.get_pool_x(), 2,
                                              target_covariate=test.covariate[:2], target_mc=2)
    assert len(indices) == len(z) == len(set(indices)) == 2
    assert np.isin(z, [0, 1]).all()
    assert (np.asarray(indices) >= 0).all()
    assert (np.asarray(indices) < data.get_pool_x().shape[0]).all()
