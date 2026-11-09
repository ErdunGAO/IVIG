import numpy as np
import pytest
from src.library.metrics import weighted_pehe_np, weighted_sqrt_pehe_np, metric_record, read_sqrt_pehe


def test_known_cate_errors_one_square_root():
    assert weighted_pehe_np([3, 4], [0, 0]) == 12.5
    assert weighted_sqrt_pehe_np([3, 4], [0, 0]) == pytest.approx(np.sqrt(12.5))
    assert weighted_sqrt_pehe_np([3, 4], [0, 0], [1, 3]) == pytest.approx(np.sqrt(14.25))
    assert weighted_sqrt_pehe_np([[3], [4]], [0, 0], [10, 30]) == pytest.approx(np.sqrt(14.25))


@pytest.mark.parametrize('pred,true,weight', [([], [], None), ([1, 2], [1], None),
    ([np.nan], [0], None), ([1, 2], [0, 0], [-1, 2]), ([1], [0], [0]),
    ([1], [0], [np.inf]), ([1, 2], [0, 0], [1])])
def test_invalid_inputs_fail(pred, true, weight):
    with pytest.raises(ValueError):
        weighted_sqrt_pehe_np(pred, true, weight)


def test_v2_and_legacy_are_rmse_without_second_root():
    row = metric_record(4)
    assert row['pehe'] == 16
    assert read_sqrt_pehe(row) == 4
    assert read_sqrt_pehe({'pehe': 4}) == 4
    assert read_sqrt_pehe({'pehe': 4, 'metric': 'sqrt_pehe', 'metric_value': 4}) == 4


@pytest.mark.parametrize('row', [dict(metric_record(4), pehe=4),
    {'metric_schema_version': 3, 'pehe': 4}, {'metric': 'target_ate_abs_error', 'pehe': 4},
    {'pehe': float('nan')}, {'pehe': -1}, dict(metric_record(4), metric_value=2)])
def test_invalid_metric_records_fail(row):
    with pytest.raises(ValueError):
        read_sqrt_pehe(row)
