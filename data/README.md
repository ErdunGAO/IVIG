# Data

The synthetic and TripAdvisor benchmarks are generated inside the code and need
no downloads.

## IHDP

The IHDP benchmark uses the standard NPCI covariate file `ihdp_npci_1.csv`
(747 rows; columns `treatment, y_factual, y_cfactual, mu0, mu1, x1..x25`), as
distributed with e.g. [CEVAE](https://github.com/AMLab-Amsterdam/CEVAE) and
[causal-bald](https://github.com/OATML/causal-bald).  Place it at

```
data/ihdp/ihdp_npci_1.csv
```

Alternatively, use `ihdp --root /path/to/directory`, or set `IVIG_IHDP_CSV`
to the full file path. An explicit `--root` takes precedence over the
environment variable.

No benchmark data or experiment results are distributed in this repository.
Obtain external data separately and follow the original provider's terms.
