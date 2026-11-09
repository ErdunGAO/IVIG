"""CATE metrics and explicit serialization of sqrt-PEHE (effect RMSE).

Historical IVIG records used the key ``pehe`` for an already rooted value.
Only this known legacy schema is supported; never take its square root again.
"""

import numpy as np

METRIC_SCHEMA_VERSION = 2


def weighted_pehe_np(pred_cate, true_cate, sample_weight=None):
    """Mean squared CATE error, with optional nonnegative target weights."""
    pred = np.asarray(pred_cate, dtype=np.float64).reshape(-1)
    true = np.asarray(true_cate, dtype=np.float64).reshape(-1)
    if pred.size == 0 or pred.shape != true.shape:
        raise ValueError("Predicted and true CATE must have equal, nonzero length.")
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError("CATE values must be finite.")
    error = np.square(pred - true)
    if not np.isfinite(error).all():
        raise ValueError("Squared CATE errors must be finite.")
    if sample_weight is None:
        return float(error.mean())
    weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if weight.shape != pred.shape or not np.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("Weights must be finite, nonnegative and match the CATE length.")
    total = float(weight.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Weights must have positive finite total mass.")
    return float(np.dot(weight / total, error))


def weighted_sqrt_pehe_np(pred_cate, true_cate, sample_weight=None):
    """Square root of the mean squared CATE error (one square root)."""
    return float(np.sqrt(weighted_pehe_np(pred_cate, true_cate, sample_weight)))


def metric_record(sqrt_pehe):
    """Serialize RMSE explicitly, alongside its squared-error counterpart."""
    value = float(sqrt_pehe)
    if not np.isfinite(value) or value < 0 or not np.isfinite(value ** 2):
        raise ValueError("sqrt_pehe must be finite and nonnegative.")
    return {"metric_schema_version": METRIC_SCHEMA_VERSION,
            "metric": "sqrt_pehe", "metric_value": value,
            "sqrt_pehe": value, "pehe": value ** 2}


def read_sqrt_pehe(row):
    """Read v2 records or the known unversioned IVIG/IV_Models legacy schema."""
    version = row.get("metric_schema_version", 1)
    if version not in (1, METRIC_SCHEMA_VERSION):
        raise ValueError(f"Unsupported metric schema version: {version}")
    name = row.get("metric", row.get("metric_name"))
    if name not in (None, "sqrt_pehe"):
        raise ValueError(f"Expected sqrt_pehe, found metric {name!r}.")
    if version == METRIC_SCHEMA_VERSION:
        if name != "sqrt_pehe" or "sqrt_pehe" not in row:
            raise ValueError("Version 2 records require metric='sqrt_pehe' and sqrt_pehe.")
        value = float(row["sqrt_pehe"])
        if "pehe" in row and not np.isclose(float(row["pehe"]), value ** 2, rtol=1e-10, atol=1e-12):
            raise ValueError("Inconsistent PEHE and sqrt-PEHE in version 2 record.")
    else:
        value = float(row["sqrt_pehe"] if "sqrt_pehe" in row else row["pehe"])
        if "pehe" in row and "sqrt_pehe" in row and not np.isclose(float(row['pehe']), value):
            raise ValueError("Inconsistent legacy metric aliases.")
    if not np.isfinite(value) or value < 0:
        raise ValueError("sqrt_pehe must be finite and nonnegative.")
    if "metric_value" in row and not np.isclose(float(row['metric_value']), value, rtol=1e-10, atol=1e-12):
        raise ValueError("Inconsistent metric_value and sqrt-PEHE.")
    return value
