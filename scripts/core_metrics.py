"""
Shared econometric logic for multigenerational housing drivers/barriers analysis.
Used identically across data sources (NHGIS, AHS, ASEC) once data is in
analysis-ready form. No SFA or tract-level frontier logic — OLS/diagnostics only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import jarque_bera
from sklearn.impute import SimpleImputer
from typing import Optional

# ── Standardized schema for analysis-ready data (unified across AHS, ASEC, NHGIS) ──
# ── Standardized schema for analysis-ready data (unified across AHS, ASEC, NHGIS) ──
ANALYSIS_READY_SCHEMA = {
    "id_cols": ["GEOID", "Area_Name", "COUNTY_GEOID", "STATEA"],  # Added for regional clustering/fixed effects
    "target_col": "Multigen_Rate",
    "weight_col": "_total_hh",  
    "feature_cols": [
        "Pct_65Plus", "Pct_Under18", "Pct_Hispanic", "Pct_Asian_NH", "Pct_Black_NH",
        "Pct_ForeignBorn", "Median_HH_Income", "Poverty_Rate", "Gini_Index", "Pct_SNAP",
        "Pct_SingleFamily", "Pct_5PlusUnits", "Pct_MobileHome", "Vacancy_Rate", "Pct_LargeUnits",
        "Median_Year_Built",
        "Pct_Owner", "Avg_HH_Size", "Pct_HighRent", "Pct_HighOwnerCost",
        "Pct_NonMultigen_FamilyHH", "Pct_MarriedCouple", "Pct_LivingAlone", "Pct_HH_With_Children",
        "Pct_BachelorPlus", "Pct_LessThanHS",
        "Pct_PublicTransit", "Pct_WorkFromHome", "Pct_NoVehicle",
        "NatWalkInd", "TransitFreq", "StreetDensity",
        "Pct_LimitedEnglish", "Pct_SameHouse1YrAgo", "Pct_NotInLaborForce", "Pct_Uninsured",
        "Lasso_1", "Lasso_2", "Lasso_3",  # Top 3 from run_lasso_feature_selection; replace with NHGIS codes from output/lasso_feature_shortlist.csv
    ],
}

FEATURE_LABELS = {
    "Pct_65Plus": "% Population 65+",
    "Pct_Under18": "% Population Under 18",
    "Pct_Hispanic": "% Hispanic/Latino",
    "Pct_Asian_NH": "% Asian (Non-Hispanic)",
    "Pct_Black_NH": "% Black (Non-Hispanic)",
    "Pct_ForeignBorn": "% Foreign-Born",
    "Median_HH_Income": "Median Household Income ($)",
    "Poverty_Rate": "Poverty Rate (%)",
    "Gini_Index": "Gini Index (Inequality)",
    "Pct_SNAP": "% Households on SNAP",
    "Pct_SingleFamily": "% Single-Family Homes",
    "Pct_5PlusUnits": "% 5+ Unit Buildings",
    "Pct_MobileHome": "% Mobile Homes",
    "Vacancy_Rate": "Housing Vacancy Rate (%)",
    "Pct_LargeUnits": "% Units with 4+ Bedrooms",
    "Median_Year_Built": "Median Year Structure Built",
    "Pct_Owner": "% Owner-Occupied",
    "Avg_HH_Size": "Average Household Size",
    "Pct_HighRent": "% High Rent ($1500+/mo)",
    "Pct_HighOwnerCost": "% High Owner Costs",
    "Pct_NonMultigen_FamilyHH": "% Non-Multigen Family HH (Purified)",
    "Pct_MarriedCouple": "% Married-Couple HH",
    "Pct_LivingAlone": "% Living Alone",
    "Pct_HH_With_Children": "% HH with Children Under 18",
    "Pct_BachelorPlus": "% Bachelor's Degree or Higher",
    "Pct_LessThanHS": "% Less Than High School",
    "Pct_PublicTransit": "% Commute by Public Transit",
    "Pct_WorkFromHome": "% Work From Home",
    "Pct_NoVehicle": "% Households No Vehicle",
    "NatWalkInd": "Walkability Index (EPA)",
    "TransitFreq": "Transit Frequency (EPA)",
    "StreetDensity": "Street Intersection Density",
    "Pct_LimitedEnglish": "% Limited English HH",
    "Pct_SameHouse1YrAgo": "% Same House 1 Year Ago",
    "Pct_NotInLaborForce": "% Not in Labor Force",
    "Pct_Uninsured": "% Uninsured",
    "Lasso_1": "Lasso discovery 1 (replace with top-3 NHGIS code from shortlist)",
    "Lasso_2": "Lasso discovery 2 (replace with top-3 NHGIS code from shortlist)",
    "Lasso_3": "Lasso discovery 3 (replace with top-3 NHGIS code from shortlist)",
}


def sig_stars(p: float) -> str:
    """Map p-value to significance stars."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "."
    return ""


def get_available_features(df: pd.DataFrame, candidate_cols: Optional[list[str]] = None) -> list[str]:
    """Return list of candidate feature columns that exist in df."""
    cols = candidate_cols or ANALYSIS_READY_SCHEMA["feature_cols"]
    return [c for c in cols if c in df.columns]


def prepare_analysis_df(
    df: pd.DataFrame,
    target_col: str = "Multigen_Rate",
    weight_col: Optional[str] = "_total_hh",
    feature_cols: Optional[list[str]] = None,
    corr_threshold: float = 0.80,
    vif_threshold: float = 10.0,
    winsorize_quantiles: tuple[float, float] = (0.01, 0.99),
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Filter, impute, winsorize, and screen features. Returns (df_ready, final_features, dropped).
    """
    feature_cols = feature_cols or get_available_features(df)
    id_cols = [c for c in ANALYSIS_READY_SCHEMA["id_cols"] if c in df.columns]
    keep = [target_col] + id_cols + feature_cols
    if weight_col and weight_col in df.columns:
        keep.append(weight_col)
    keep = [c for c in keep if c in df.columns]
    work = df[keep].copy()

    if weight_col and weight_col in work.columns:
        work = work[work[weight_col].fillna(0) > 0]
    work = work[work[target_col].notna()].copy()
    work = work.reset_index(drop=True)

    for col in feature_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    imputer = SimpleImputer(strategy="median")
    work[feature_cols] = imputer.fit_transform(work[feature_cols])

    for col in feature_cols:
        q1, q99 = work[col].quantile(winsorize_quantiles[0]), work[col].quantile(winsorize_quantiles[1])
        work[col] = work[col].clip(q1, q99)
    y_q1, y_q99 = work[target_col].quantile(winsorize_quantiles[0]), work[target_col].quantile(winsorize_quantiles[1])
    work[target_col] = work[target_col].clip(y_q1, y_q99)

    # Correlation screen
    X_raw = work[feature_cols]
    corr_matrix = X_raw.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    dropped_corr = []
    for column in upper.columns:
        if any(upper[column] > corr_threshold):
            dropped_corr.append(column)
            upper = upper.drop(index=column, columns=column, errors="ignore")
    retained = [c for c in feature_cols if c not in dropped_corr]

    # VIF screen
    final_features = retained.copy()
    while True:
        X_const = sm.add_constant(work[final_features])
        vifs = [variance_inflation_factor(X_const.values, i + 1) for i in range(len(final_features))]
        max_vif = max(vifs)
        if max_vif <= vif_threshold:
            break
        worst_idx = vifs.index(max_vif)
        final_features.pop(worst_idx)

    dropped = [c for c in feature_cols if c not in final_features]
    return work, final_features, dropped


def run_ols_pipeline(
    df: pd.DataFrame,
    target_col: str = "Multigen_Rate",
    weight_col: Optional[str] = "_total_hh",
    feature_cols: Optional[list[str]] = None,
) -> dict:
    """
    Run OLS with diagnostics and County-Clustered Robust Standard Errors.
    """
    work, final_features, _ = prepare_analysis_df(
        df, target_col=target_col, weight_col=weight_col, feature_cols=feature_cols
    )
    y = work[target_col]
    X = sm.add_constant(work[final_features])

    # County-clustered SEs when COUNTY_GEOID present (addresses spatial dependency)
    use_cluster = (
        "COUNTY_GEOID" in work.columns
        and work["COUNTY_GEOID"].notna().all()
        and work["COUNTY_GEOID"].astype(str).str.len().gt(0).all()
    )
    cov_spec = (
        {"cov_type": "cluster", "cov_kwds": {"groups": work["COUNTY_GEOID"]}}
        if use_cluster
        else {"cov_type": "HC3"}
    )

    # 1. Baseline Models
    ols_model = sm.OLS(y, X).fit()

    # 2. County-Clustered (or HC3) Robust Model
    ols_robust = sm.OLS(y, X).fit(**cov_spec)

    wls_model = None
    if weight_col and weight_col in work.columns:
        weights = work[weight_col].clip(lower=1)
        wls_model = sm.WLS(y, X, weights=weights).fit(**cov_spec)

    y_log = np.log(y + 0.01)
    ols_log = sm.OLS(y_log, X).fit(**cov_spec)

    # ... Diagnostics (Same as before) ...
    resid = ols_model.resid
    bp_lm, bp_pval, _, _ = het_breuschpagan(resid, X)
    rng = np.random.RandomState(42)
    sub_idx = rng.choice(len(resid), size=min(10000, len(resid)), replace=False)
    try:
        wh_lm, wh_pval, _, _ = het_white(resid.iloc[sub_idx], X.iloc[sub_idx])
    except Exception:
        wh_lm, wh_pval = np.nan, np.nan
    jb_stat, jb_pval, jb_skew, jb_kurtosis = jarque_bera(resid)
    
    # Standard Reset Test logic
    from scipy.stats import f as f_dist
    y_hat = ols_model.fittedvalues
    X_reset = X.copy()
    X_reset["y_hat_sq"] = y_hat ** 2
    X_reset["y_hat_cu"] = y_hat ** 3
    reset_model = sm.OLS(y, X_reset).fit()
    r_unres, r_res = reset_model.rsquared, ols_model.rsquared
    k_unres, k_res = reset_model.df_model, ols_model.df_model
    n = len(y)
    reset_fstat = ((r_unres - r_res) / (k_unres - k_res)) / ((1 - r_unres) / (n - k_unres - 1))
    reset_pval = 1 - f_dist.cdf(reset_fstat, k_unres - k_res, n - k_unres - 1)

    labels = {f: FEATURE_LABELS.get(f, f) for f in final_features}

    coef_table = pd.DataFrame({
        "Feature": final_features,
        "Label": [labels.get(f, f) for f in final_features],
        "OLS_Coef": [ols_model.params.get(f, np.nan) for f in final_features],
        "OLS_SE": [ols_model.bse.get(f, np.nan) for f in final_features],
        "OLS_pval": [ols_model.pvalues.get(f, np.nan) for f in final_features],
        "Robust_Coef": [ols_robust.params.get(f, np.nan) for f in final_features],
        "Robust_SE": [ols_robust.bse.get(f, np.nan) for f in final_features],
        "Robust_pval": [ols_robust.pvalues.get(f, np.nan) for f in final_features],
    })
    if wls_model is not None:
        coef_table["WLS_Coef"] = [wls_model.params.get(f, np.nan) for f in final_features]
        coef_table["WLS_SE"] = [wls_model.bse.get(f, np.nan) for f in final_features]
        coef_table["WLS_pval"] = [wls_model.pvalues.get(f, np.nan) for f in final_features]
    coef_table["Significance"] = coef_table["Robust_pval"].apply(sig_stars)
    coef_table["Abs_Coef"] = coef_table["Robust_Coef"].abs()
    coef_table = coef_table.sort_values("Abs_Coef", ascending=False)

    X_std = (work[final_features] - work[final_features].mean()) / work[final_features].std()
    y_std = (y - y.mean()) / y.std()
    X_std_c = sm.add_constant(X_std)
    beta_model = sm.OLS(y_std, X_std_c).fit(**cov_spec)
    beta_table = pd.DataFrame({
        "Feature": final_features,
        "Label": [labels.get(f, f) for f in final_features],
        "Beta_Coef": [beta_model.params.get(f, np.nan) for f in final_features],
        "Beta_SE": [beta_model.bse.get(f, np.nan) for f in final_features],
        "Beta_pval": [beta_model.pvalues.get(f, np.nan) for f in final_features],
        "Significance": [sig_stars(beta_model.pvalues.get(f, 1)) for f in final_features],
    })
    beta_table["Abs_Beta"] = beta_table["Beta_Coef"].abs()
    beta_table = beta_table.sort_values("Abs_Beta", ascending=False)

    return {
        "ols_model": ols_model,
        "ols_robust": ols_robust,
        "wls_model": wls_model,
        "ols_log": ols_log,
        "coef_table": coef_table,
        "beta_table": beta_table,
        "diagnostics": {
            "breusch_pagan": (bp_lm, bp_pval),
            "white": (wh_lm, wh_pval),
            "jarque_bera": (jb_stat, jb_pval, jb_skew, jb_kurtosis),
            "reset": (reset_fstat, reset_pval),
        },
        "final_features": final_features,
        "feature_labels": labels,
        "work_df": work,
        "y": y,
        "X": X,
    }


def interpret_effect(
    label: str,
    feature_name: str,
    unstd_coef: float,
    beta: float,
    sig: str,
    mean_target: float = 5.6,
) -> list[str]:
    """Plain-English interpretation lines for report (no SFA)."""
    lines = []
    abs_coef = abs(unstd_coef)
    direction = "higher" if unstd_coef > 0 else "lower"
    rel_sign = "+" if unstd_coef > 0 else "-"
    is_pct_var = label.startswith("%") or "Rate" in label
    is_income = "Income" in label
    is_gini = "Gini" in label
    is_year = "Year" in label
    is_walkability = "Walkability" in label

    lines.append(f"  - {label}  [{sig}]")
    if is_income:
        change_amt = abs_coef * 10000
        rel_pct = (change_amt / mean_target) * 100
        lines.append(f"    A $10,000 increase in median household income is associated with a {change_amt:.2f} pp {direction} multigenerational rate (absolute), equivalent to a {rel_sign}{rel_pct:.1f}% relative change from mean {mean_target:.1f}%.")
    elif is_gini:
        change_amt = abs_coef * 0.10
        rel_pct = (change_amt / mean_target) * 100
        lines.append(f"    A 0.10-point increase in the Gini index is associated with a {change_amt:.2f} pp {direction} multigenerational rate (absolute), equivalent to a {rel_sign}{rel_pct:.1f}% relative change from mean {mean_target:.1f}%.")
    elif is_year:
        change_amt = abs_coef * 10
        rel_pct = (change_amt / mean_target) * 100
        lines.append(f"    A 10-year increase in median year built is associated with a {change_amt:.3f} pp {direction} multigenerational rate (absolute), equivalent to a {rel_sign}{rel_pct:.1f}% relative change from mean {mean_target:.1f}%.")
    elif is_walkability:
        change_amt = abs_coef * 5
        rel_pct = (change_amt / mean_target) * 100
        lines.append(f"    A 5-point increase in the EPA Walkability Index is associated with a {change_amt:.2f} pp {direction} multigenerational rate (absolute), equivalent to a {rel_sign}{rel_pct:.1f}% relative change from mean {mean_target:.1f}%.")
    elif is_pct_var:
        change_10 = abs_coef * 10
        rel_pct = (change_10 / mean_target) * 100
        lines.append(f"    A 10 pp increase in {label.lower()} is associated with a {change_10:.2f} pp {direction} multigenerational rate (absolute), equivalent to a {rel_sign}{rel_pct:.1f}% relative change from mean {mean_target:.1f}%.")
    else:
        change_10 = abs_coef * 10
        rel_pct = (change_10 / mean_target) * 100
        lines.append(f"    A 10-unit increase in {label.lower()} is associated with a {change_10:.2f} pp {direction} multigenerational rate (absolute), equivalent to a {rel_sign}{rel_pct:.1f}% relative change from mean {mean_target:.1f}%.")
    lines.append(f"    (Standardized effect: beta = {beta:+.3f}.)")
    lines.append("")
    return lines
