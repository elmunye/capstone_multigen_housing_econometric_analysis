# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Python data science project for multigenerational housing econometric analysis. Pure offline computational — no Docker, databases, APIs, or web servers required. All execution is through Jupyter notebooks and Python scripts.

### Running the application

Start JupyterLab:
```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --ServerApp.token='' --ServerApp.password=''
```

### Key services

| Service | Command | Notes |
|---------|---------|-------|
| JupyterLab | `jupyter lab --port=8888` | Primary dev interface; see README for notebook execution order |

### Linting

```bash
flake8 scripts/ --max-line-length=120
```

No `.flake8` config file exists; use `--max-line-length=120` to match the codebase style. Pre-existing lint warnings (trailing whitespace, long lines in docstrings) are intentional — do not fix unless asked.

### Module imports from notebooks

Notebooks use `sys.path.insert(0, ...)` to add the `scripts/` directory. When running scripts directly from the repo root, use:
```python
import sys; sys.path.insert(0, 'scripts')
```
`scripts/advanced_metrics.py` and `scripts/ingest_nhgis.py` use bare `from core_metrics import ...` (not `from scripts.core_metrics`), so they must be run with `scripts/` on `sys.path`.

### Data files

The `data/` directory is gitignored (large census datasets). Notebooks will fail at the data-loading step without these files, but all code imports and the econometric pipeline logic can be verified with synthetic data. See `README.md` for data dependency details.

### Available pipelines

- **OLS pipeline**: `core_metrics.run_ols_pipeline(df)` — baseline OLS, HC3/clustered robust SE, WLS, log-level, diagnostics
- **Advanced pipelines** (in `advanced_metrics.py`): `run_gam_pipeline(df)`, `run_xgboost_shap_pipeline(df)`, `run_quantile_pipeline(df)` — all accept a DataFrame with the analysis-ready schema
- **Lasso feature selection**: `run_lasso_feature_selection(data_dir)` — requires raw NHGIS data files in `data/raw/nhgis/`
