import json
from pathlib import Path
import numpy as np
import pytest
from src.application.workflows.evaluation import pehe, plot_convergence
from src.library.metrics import metric_record


def trial(root, name, grid, values, legacy=False):
    folder = root / name
    folder.mkdir(parents=True)
    rows = [dict(n_samples=n, **({'pehe': v} if legacy else metric_record(v))) for n, v in zip(grid, values)]
    (folder / 'active_results.json').write_text(json.dumps(rows))
    (folder / 'config.json').write_text(json.dumps({'acquisition_function': 'ivig'}))


def test_align_by_sample_count_and_average_trial_rmse(tmp_path):
    root = tmp_path / 'ivig'
    trial(root, 'trial-000', [20, 40, 60], [2, 4, 8], legacy=True)
    trial(root, 'trial-001', [20, 60], [4, 12])
    summary = pehe(root, tmp_path / 'results')
    assert summary['n_samples'] == [20, 60]
    assert summary['sqrt_pehe_mean'] == [3, 10]
    assert summary['sqrt_pehe_sem'] == pytest.approx([1, 2])
    assert summary['final_sqrt_pehe'] == 10
    assert summary['final_sqrt_pehe'] != np.sqrt((8**2 + 12**2)/2)
    plot_convergence(tmp_path / 'results', ['ivig'])
    assert (tmp_path / 'results' / 'convergence.pdf').stat().st_size > 0


def test_no_shared_samples_fails(tmp_path):
    trial(tmp_path / 'ivig', 'trial-000', [20], [2])
    trial(tmp_path / 'ivig', 'trial-001', [40], [4])
    with pytest.raises(ValueError, match='no shared'):
        pehe(tmp_path / 'ivig', tmp_path / 'results')


def test_single_trial_has_zero_sem(tmp_path):
    trial(tmp_path / 'ivig', 'trial-000', [20], [4], legacy=True)
    s = pehe(tmp_path / 'ivig', tmp_path / 'results')
    assert s['sqrt_pehe_mean'] == [4]
    assert s['sqrt_pehe_sem'] == [0]


def test_empty_plot_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        plot_convergence(tmp_path, ['ivig'])


@pytest.mark.parametrize('band,scale', [('std', np.sqrt(2)), ('sem', 1)])
def test_plot_bands_preserve_rooted_mean(tmp_path, monkeypatch, band, scale):
    import matplotlib.axes
    trial(tmp_path / 'ivig', 'trial-000', [20, 40], [2, 4])
    trial(tmp_path / 'ivig', 'trial-001', [20, 40], [4, 8])
    pehe(tmp_path / 'ivig', tmp_path / 'results')
    bands = []
    original = matplotlib.axes.Axes.fill_between
    def record(ax, x, low, high, **kwargs):
        bands.append((low.copy(), high.copy()))
        return original(ax, x, low, high, **kwargs)
    monkeypatch.setattr(matplotlib.axes.Axes, 'fill_between', record)
    options = {} if band == 'std' else {'error_band': band}
    plot_convergence(tmp_path / 'results', ['ivig'], **options)
    np.testing.assert_allclose(bands[0][0], [3, 6] - scale * np.array([1, 2]))
    np.testing.assert_allclose(bands[0][1], [3, 6] + scale * np.array([1, 2]))


def test_legacy_plot_can_recover_std(tmp_path):
    summary = dict(n_samples=[20], pehe_mean=[4], pehe_sem=[1], num_trials=4)
    (tmp_path / 'ivig_pehe.json').write_text(json.dumps(summary))
    plot_convergence(tmp_path, ['ivig'])
    del summary['num_trials']
    (tmp_path / 'ivig_pehe.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='Cannot recover std'):
        plot_convergence(tmp_path, ['ivig'])
    plot_convergence(tmp_path, ['ivig'], error_band='sem')
