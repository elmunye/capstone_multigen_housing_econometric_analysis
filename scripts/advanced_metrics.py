"""
Advanced/Non-linear econometric logic for multigenerational housing analysis.
Includes Lasso feature discovery, GAM, XGBoost with SHAP, and Quantile Regression.
"""

from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
import shap
from pygam import LinearGAM, s, f
from typing import Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.impute import SimpleImputer

# Import the shared preparation logic to guarantee apples-to-apples comparison with OLS
from core_metrics import prepare_analysis_df, FEATURE_LABELS

# NHGIS time-series / codebook mapping for Lasso shortlist (prioritize B25014, B25070)
NHGIS_CODEBOOK = {
    "AU08": "B25014 — Occupants per room (Overcrowding)",
    "AU2Q": "B25070 — Gross rent as % of income (Rent squeeze)",
    "AU40": "B25044 — Vehicles available",
    "AU46": "B11017 — Household type (incl. multigenerational)",
    "AUOVE": "B01001 — Sex by age",
    "AUPFE": "B02001 — Race",
    "AUYOE": "B05002 — Nativity",
    "AURUE": "B19013 — Median household income",
    "AVA1": "B19083 — Gini index",
    "AURRE": "B17001 — Poverty",
    "AUVLE": "B25024 — Units in structure",
    "AUVUE": "B25004 — Vacancy status",
    "AUVRE": "B25042 — Bedrooms",
    "AUVPE": "B25035 — Median year built",
    "AUUEE": "B25003 — Tenure",
    "AUUGE": "B25010 — Average household size",
    "AUP1": "B11001 — Household type (family)",
    "AUQNE": "B11005 — Households with under-18",
    "AUQ8": "B15003 — Educational attainment",
    "AUPWE": "B08301 — Commute",
    "AVFIE": "B25063 — Gross rent",
    "AVFSE": "B25091 — Owner costs",
    "AUTWE": "B23025 — Employment status",
    "AUTPE": "B22001 — SNAP",
    "AURLE": "B16008 — Language",
    "AUPHE": "B07001 — Geographic mobility",
    "AVH7": "B27001 — Health insurance",
}


def _nhgis_table_for_code(code: str) -> str:
    """Map NHGIS code (e.g. AU46E002) to codebook table description."""
    # NHGIS codes are like AU46E002: alpha prefix + E + cell number
    match = re.match(r"^([A-Za-z]+\d*)", str(code))
    prefix = (match.group(1) if match else code)[:5]
    for key in sorted(NHGIS_CODEBOOK.keys(), key=len, reverse=True):
        if prefix.startswith(key) or key in prefix:
            return NHGIS_CODEBOOK[key]
    return "— (see NHGIS codebook)"


def run_lasso_feature_selection(
    data_dir: str,
    output_path: str = "output/lasso_feature_shortlist.csv",
    target_col: str = "Multigen_Rate",
    top_corr_n: int = 100,
    top_nonzero_n: int = 30,
    n_alphas: int = 100,
    cv: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Lasso-based discovery of high-impact omitted variables from raw NHGIS columns.
    Reads only from data_dir (e.g. data/raw/nhgis); never modifies source CSVs.
    Steps: load raw wide data → correlation filter (top 100) → LassoCV (standardized)
    → shortlist top 30 non-zero coefficients → write output_path.
    """
    from ingest_nhgis import load_raw_nhgis_wide

    geo_id_cols = {
        "GISJOIN", "YEAR", "STUSAB", "STATE", "COUNTY", "GEO_ID", "TL_GEO_ID",
        "NAME_E", "NAME_M", "REGIONA", "DIVISIONA", "STATEA", "COUNTYA", "TRACTA",
        "BLKGRPA", "COUSUBA", "PLACEA", "CONCITA", "AIANHHA", "RES_ONLYA", "TRUSTA",
        "AIHHTLI", "AITSA", "ANRCA", "CBSAA", "CSAA", "METDIVA", "UAA", "CDCURRA",
        "SLDUA", "SLDLA", "ZCTA5A", "SUBMCDA", "SDELMA", "SDSECA", "SDUNIA",
        "PCI", "PUMAA", "BTTRA", "BTBGA", "GEOID", "COUNTY_GEOID",
    }

    df, codebook = load_raw_nhgis_wide(data_dir)
    if target_col not in df.columns:
        raise ValueError(f"Target {target_col} not in raw data. Ensure AU46E001/AU46E002 are present.")
    df = df.dropna(subset=[target_col]).copy()

    candidate_cols = [
        c for c in df.columns
        if c not in geo_id_cols and c != target_col
        and df[c].dtype in (np.floating, np.integer, "int64", "float64")
    ]
    # Drop columns that are all NaN or zero variance
    usable = [c for c in candidate_cols if df[c].notna().sum() > 100 and df[c].std(skipna=True) > 0]
    if len(usable) < top_corr_n:
        top_corr_n = min(top_corr_n, len(usable))

    corr_with_target = df[usable].corrwith(df[target_col]).abs().sort_values(ascending=False)
    top_corr_cols = corr_with_target.head(top_corr_n).index.tolist()

    X_raw = df[top_corr_cols].copy()
    y = df[target_col].values
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X_raw)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_imp)

    lasso = LassoCV(n_alphas=n_alphas, cv=cv, random_state=random_state)
    lasso.fit(X_std, y)
    coefs = pd.Series(lasso.coef_, index=top_corr_cols)
    non_zero = coefs[coefs != 0]
    non_zero = non_zero.reindex(non_zero.abs().sort_values(ascending=False).index)
    shortlist_cols = non_zero.head(top_nonzero_n).index.tolist()
    shortlist_coefs = non_zero.head(top_nonzero_n).values

    shortlist = pd.DataFrame({
        "nhgis_code": shortlist_cols,
        "standardized_coef": shortlist_coefs,
        "abs_coef": np.abs(shortlist_coefs),
        "codebook_table": [codebook.get(c, "— (see NHGIS codebook)") for c in shortlist_cols],
    })
    shortlist = shortlist.sort_values("abs_coef", ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    shortlist.to_csv(output_path, index=False)

    return {
        "shortlist": shortlist,
        "shortlist_codes": shortlist_cols,
        "lasso_model": lasso,
        "scaler": scaler,
        "imputer": imp,
        "top_corr_cols": top_corr_cols,
        "optimal_alpha": lasso.alpha_,
        "output_path": output_path,
    }

def run_quantile_pipeline(
    df: pd.DataFrame,
    target_col: str = "Multigen_Rate",
    feature_cols: Optional[list[str]] = None,
    quantiles: list[float] = [0.1, 0.5, 0.9]
) -> dict:
    """
    Runs Quantile Regression to see how determinants change at different
    concentrations of multigenerational housing (e.g., 10th vs 90th percentile).
    """
    work, final_features, _ = prepare_analysis_df(df, target_col=target_col, feature_cols=feature_cols)
    y = work[target_col]
    X = sm.add_constant(work[final_features])
    
    models = {}
    for q in quantiles:
        # max_iter increased for convergence on tricky census datasets
        mod = sm.QuantReg(y, X).fit(q=q, max_iter=2000)
        models[f"q_{q}"] = mod
        
    return {
        "models": models,
        "final_features": final_features,
        "X": X, "y": y
    }

def run_xgboost_shap_pipeline(
    df: pd.DataFrame,
    target_col: str = "Multigen_Rate",
    feature_cols: Optional[list[str]] = None,
) -> dict:
    """
    Fits an XGBoost Regressor and calculates SHAP values to capture 
    complex non-linearities and feature interactions automatically.
    """
    work, final_features, _ = prepare_analysis_df(df, target_col=target_col, feature_cols=feature_cols)
    y = work[target_col]
    X = work[final_features] # No constant needed for trees
    
    # Baseline XGBoost with conservative hyperparameters to prevent extreme overfitting
    model = xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X, y)
    
    # Explain the model's predictions using SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    
    return {
        "model": model,
        "explainer": explainer,
        "shap_values": shap_values,
        "final_features": final_features,
        "X": X, "y": y
    }

def run_gam_pipeline(
    df: pd.DataFrame,
    target_col: str = "Multigen_Rate",
    feature_cols: Optional[list[str]] = None,
    n_splines: int = 10
) -> dict:
    """
    Fits a Generalized Additive Model (GAM) to explicitly map the non-linear
    curves of each determinant.
    """
    work, final_features, _ = prepare_analysis_df(df, target_col=target_col, feature_cols=feature_cols)
    y = work[target_col].values
    X = work[final_features].values
    
    # Build a GAM where every feature gets a smooth spline (s)
    # n_splines limits the "wiggliness" to prevent overfitting
    gam = LinearGAM(n_splines=n_splines)
    gam.gridsearch(X, y, progress=False)
    
    return {
        "model": gam,
        "final_features": final_features,
        "X": work[final_features], 
        "y": work[target_col]
    }

def generate_systematic_comparison(
    ols_results: dict, 
    xgb_results: dict, 
    output_path: str = None
) -> pd.DataFrame:
    """
    Systematically merges OLS results with XGBoost SHAP importance and 
    approximates 'SHAP Elasticities' for easier reading.
    """
    # 1. Extract OLS stats
    ols_df = ols_results["coef_table"][["Feature", "Label", "Robust_Coef", "Robust_pval", "Significance"]].copy()
    ols_df.columns = ["Feature", "Label", "OLS_Coef", "OLS_p", "OLS_Sig"]

    # 2. Extract XGBoost SHAP Importance
    shap_values = xgb_results["shap_values"]
    X = xgb_results["X"]
    
    # Mean Absolute SHAP (Global Importance)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    
    # 3. Approximate 'SHAP Elasticity' (Mean % impact on target for a 1% change in feature)
    # This represents the average sensitivity of the model to that feature.
    target_mean = xgb_results["y"].mean()
    feature_means = X.mean()
    
    # Elasticity proxy: (Mean SHAP / Target Mean) / (Standard Deviation Feature / Feature Mean)
    # This scales the SHAP impact into a pseudo-elasticity metric
    shap_elasticity = (mean_abs_shap / target_mean) / (X.std() / feature_means)
    
    xgb_stats = pd.DataFrame({
        "Feature": xgb_results["final_features"],
        "SHAP_Importance": mean_abs_shap,
        "SHAP_Elasticity": shap_elasticity
    })

    # 4. Merge
    comparison = pd.merge(ols_df, xgb_stats, on="Feature", how="inner")
    
    # 5. Flag Non-Linearity
    # If a feature has high SHAP importance but low OLS significance, it's likely non-linear.
    comparison["Potential_NonLinearity"] = (comparison["OLS_p"] > 0.05) & (comparison["SHAP_Importance"] > comparison["SHAP_Importance"].median())

    comparison = comparison.sort_values("SHAP_Importance", ascending=False)
    
    if output_path:
        # This ensures the directory is created only if it doesn't exist,
        # using the path provided by the notebook.
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create directory if there's a directory path
            os.makedirs(output_dir, exist_ok=True)
        comparison.to_csv(output_path, index=False)
        print(f"File saved to: {output_path}")
        
    return comparison

def analyze_gam_nonlinearities(gam_results: dict) -> pd.DataFrame:
    """
    Quantifies the non-linear curves from the GAM model.
    Identifies if a feature is 'Increasing', 'Decreasing', or 'U-Shaped'.
    """
    gam = gam_results["model"]
    features = gam_results["final_features"]
    
    analysis_rows = []
    
    for i, col in enumerate(features):
        # Generate grid and partial dependence
        XX = gam.generate_X_grid(term=i)
        pdep_result = gam.partial_dependence(term=i, X=XX)
        
        # Handle different return formats from partial_dependence
        if isinstance(pdep_result, tuple):
            pdep = pdep_result[0]
        else:
            pdep = pdep_result
        
        # Ensure pdep is 1D array
        pdep = np.asarray(pdep).flatten()
        
        # Extract the feature values for this term
        # XX from generate_X_grid(term=i) has shape (n_points, n_features)
        # We want the values of feature i across the grid
        if XX.ndim > 1:
            feature_values = XX[:, i]
        else:
            feature_values = XX
        feature_values = np.asarray(feature_values).flatten()
        
        # Ensure pdep and feature_values have the same length
        if len(pdep) != len(feature_values):
            # If lengths don't match, use indices instead
            feature_values = np.arange(len(pdep))
        
        # Calculate local slopes (derivatives) to find thresholds
        slopes = np.gradient(pdep, feature_values)
        
        # Ensure slopes is 1D (np.gradient can return tuple for multi-dim)
        if isinstance(slopes, tuple):
            slopes = slopes[0]
        slopes = np.asarray(slopes).flatten()
        
        analysis_rows.append({
            "Feature": col,
            "Label": FEATURE_LABELS.get(col, col),
            "Max_Effect": np.max(pdep) - np.min(pdep),
            "Start_Slope": slopes[0] if len(slopes) > 0 else 0.0,
            "End_Slope": slopes[-1] if len(slopes) > 0 else 0.0,
            "Is_NonLinear": np.std(slopes) > 0.01, # Threshold for 'wiggliness'
            "Trend": "Non-monotonic/U-shaped" if np.any(slopes > 0) and np.any(slopes < 0) else "Monotonic"
        })
        
    return pd.DataFrame(analysis_rows).sort_values("Max_Effect", ascending=False)


def compare_model_performance(ols_results: dict, xgb_results: dict) -> pd.DataFrame:
    """
    Compares the predictive accuracy of OLS vs XGBoost.
    """
    # 1. Get Actuals and Predictions
    y_true = ols_results["y"]
    
    # OLS Predictions
    y_pred_ols = ols_results["ols_model"].predict(ols_results["X"])
    
    # XGBoost Predictions
    y_pred_xgb = xgb_results["model"].predict(xgb_results["X"])
    
    # 2. Calculate Metrics
    perf_data = []
    for name, y_pred in [("OLS (Baseline)", y_pred_ols), ("XGBoost (Advanced)", y_pred_xgb)]:
        perf_data.append({
            "Model": name,
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R-Squared": r2_score(y_true, y_pred)
        })
        
    return pd.DataFrame(perf_data)