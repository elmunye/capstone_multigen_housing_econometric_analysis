"""
Advanced/Non-linear econometric logic for multigenerational housing analysis.
Includes Generalized Additive Models (GAM), XGBoost with SHAP, and Quantile Regression.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
import shap
from pygam import LinearGAM, s, f
from typing import Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Import the shared preparation logic to guarantee apples-to-apples comparison with OLS
from core_metrics import prepare_analysis_df, FEATURE_LABELS

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