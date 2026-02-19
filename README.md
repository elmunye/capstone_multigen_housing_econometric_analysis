# capstone_multigen_housing_econometric_analysis

Drivers and barriers of multigenerational living: unified econometric modeling across multiple data sources. **No SFA** (Stochastic Frontier Analysis); that work lives in the sister repo `capstone_multigen_sfa_market_gap_analysis`.

## Focus

- **Econometric models:** OLS (baseline + HC3 robust SE), WLS, log-level robustness, VIF/diagnostics (Breusch-Pagan, White, Jarque-Bera, Ramsey RESET).
- **Modular pattern:** Data ingestion is **unique per source** (AHS, ASEC, NHGIS); econometric modeling is **identical** via `scripts/core_metrics.py`.

## Data Dependencies

- **NHGIS** (ACS tract-level)
- **EPA Smart Location Database**
- **AHS** (American Housing Survey)
- **ASEC** (CPS Annual Social and Economic Supplement)

## Directory Structure

```
capstone_multigen_housing_econometric_analysis/
├── data/
│   ├── raw/          # Subfolders: nhgis, smart_location, ahs, asec
│   └── processed/    # Analysis-ready CSVs (standardized column names)
├── scripts/
│   ├── core_metrics.py   # Shared econometric logic (OLS, diagnostics, schema)
│   └── ingest_nhgis.py   # NHGIS + SLD → analysis-ready
├── notebooks/
│   ├── 01_ingestion_NHGIS.ipynb
│   ├── 01_ingestion_AHS.ipynb
│   ├── 01_ingestion_ASEC.ipynb
│   ├── 02_analysis_NHGIS.ipynb
│   ├── 02_analysis_AHS.ipynb
│   ├── 02_analysis_ASEC.ipynb
│   └── 03_comparative_master.ipynb   # Compare results across sources
├── requirements.txt
└── README.md
```

## Analysis-Ready Schema

Ingestion notebooks must produce DataFrames with **identical column naming** so that `core_metrics.run_ols_pipeline()` works for every source:

- **Target:** `Multigen_Rate`
- **Optional weight:** `_total_hh`
- **IDs:** `GEOID`, `Area_Name` (or source-specific id)
- **Features:** See `scripts/core_metrics.ANALYSIS_READY_SCHEMA["feature_cols"]` (e.g. `Pct_65Plus`, `Median_HH_Income`, `Pct_Owner`, `NatWalkInd`, …).

## Usage

### NHGIS Analysis (Complete)

The NHGIS analysis is **fully implemented** and ready to run:

1. **Ingestion:** Run `01_ingestion_NHGIS.ipynb` to load NHGIS CSVs and Smart Location Database, engineer features, and save to `data/processed/nhgis_analysis_ready.csv`.
2. **Analysis:** Run `02_analysis_NHGIS.ipynb` to run the full econometric pipeline (OLS, diagnostics, visualizations) and save results to `output/`.

**Data files:** NHGIS CSVs and Smart Location Database are in `data/raw/nhgis/` and `data/raw/smart_location/`.

### AHS and ASEC (To Be Completed)

1. **Ingestion:** Implement `01_ingestion_AHS.ipynb` and `01_ingestion_ASEC.ipynb` to map AHS/ASEC variables to the analysis-ready schema.
2. **Analysis:** Run `02_analysis_AHS.ipynb` and `02_analysis_ASEC.ipynb` (these will work once ingestion is complete).
3. **Comparison:** Run `03_comparative_master.ipynb` to compare results across all three sources.

### Quick Start

From repo root:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

Start with `01_ingestion_NHGIS.ipynb` → `02_analysis_NHGIS.ipynb` to see the complete NHGIS analysis.

## Sister Repository

- **capstone_multigen_sfa_market_gap_analysis:** Tract-level Stochastic Frontier Analysis (demand–supply gap, Moran's I, choropleths). Use that repo for SFA; this repo is econometric drivers/barriers only.
