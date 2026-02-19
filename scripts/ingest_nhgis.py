"""
NHGIS + Smart Location ingestion: build analysis-ready dataframe with standardized
column names (ANALYSIS_READY_SCHEMA). No SFA; output is for OLS/diagnostics only.
"""
import os
import glob as globmod
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from core_metrics import ANALYSIS_READY_SCHEMA, get_available_features

def parse_nhgis(filepath: str) -> pd.DataFrame:
    """Parse NHGIS CSV with 2-row header; return DataFrame with short codes (no rename)."""
    df = pd.read_csv(filepath, header=0, skiprows=[1], low_memory=False)
    return df


def load_nhgis_codebook(data_dir: str) -> tuple[dict[str, str], set[str]]:
    """
    Read the first two rows of each NHGIS CSV in data_dir/raw/nhgis:
    row 0 = column names (NHGIS codes), row 1 = variable descriptions.
    Returns (codebook, exclude_set):
      - codebook: dict mapping each column name to its description (row 2 text).
      - exclude_set: column names whose description starts with 'Margins of error' (to exclude from Lasso).
    """
    raw_nhgis = os.path.join(data_dir, "raw", "nhgis")
    if not os.path.isdir(raw_nhgis):
        raw_nhgis = data_dir
    nhgis_pattern = os.path.join(raw_nhgis, "nhgis*.csv")
    nhgis_files = sorted(globmod.glob(nhgis_pattern))
    if not nhgis_files:
        raise FileNotFoundError(f"No NHGIS files found: {nhgis_pattern}")

    codebook: dict[str, str] = {}
    exclude_set: set[str] = set()
    for filepath in nhgis_files:
        head = pd.read_csv(filepath, header=None, nrows=2, low_memory=False)
        if head.shape[0] < 2:
            continue
        names = head.iloc[0].astype(str).tolist()
        descs = head.iloc[1].tolist()
        n = min(len(names), len(descs))
        for i in range(n):
            col = names[i]
            desc = descs[i] if pd.notna(descs[i]) else ""
            desc_str = str(desc).strip()
            if col not in codebook:
                codebook[col] = desc_str
            if desc_str.lower().startswith("margins of error"):
                exclude_set.add(col)

    return codebook, exclude_set


def load_raw_nhgis_wide(data_dir: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Load all raw NHGIS CSVs from data_dir (e.g. data/raw/nhgis), merge on GISJOIN,
    coerce numeric, build GEOID/COUNTY_GEOID and household-level Multigen_Rate.
    Excludes columns whose CSV row-2 description starts with 'Margins of error'.
    Does not modify any source CSV.
    Returns (df, codebook): wide dataframe for Lasso screening and code -> description map for codebook_table.
    """
    codebook, exclude_set = load_nhgis_codebook(data_dir)

    raw_nhgis = os.path.join(data_dir, "raw", "nhgis")
    if not os.path.isdir(raw_nhgis):
        raw_nhgis = data_dir
    nhgis_pattern = os.path.join(raw_nhgis, "nhgis*.csv")
    nhgis_files = sorted(globmod.glob(nhgis_pattern))
    if not nhgis_files:
        raise FileNotFoundError(f"No NHGIS files found: {nhgis_pattern}")

    frames = [parse_nhgis(f) for f in nhgis_files]
    ns = frames[0]
    for extra in frames[1:]:
        new_cols = [c for c in extra.columns if c not in ns.columns]
        merge_keys = ["GISJOIN"] if "GISJOIN" in extra.columns else [ns.columns[0]]
        ns = ns.merge(extra[merge_keys + new_cols], on=merge_keys, how="outer")

    # Drop columns described as "Margins of error" so they are not used in Lasso
    cols_to_drop = [c for c in ns.columns if c in exclude_set]
    ns = ns.drop(columns=cols_to_drop, errors="ignore")

    geo_id_cols = {
        "GISJOIN", "YEAR", "STUSAB", "STATE", "COUNTY", "GEO_ID", "TL_GEO_ID",
        "NAME_E", "NAME_M", "REGIONA", "DIVISIONA", "STATEA", "COUNTYA", "TRACTA",
        "BLKGRPA", "COUSUBA", "PLACEA", "CONCITA", "AIANHHA", "RES_ONLYA", "TRUSTA",
        "AIHHTLI", "AITSA", "ANRCA", "CBSAA", "CSAA", "METDIVA", "UAA", "CDCURRA",
        "SLDUA", "SLDLA", "ZCTA5A", "SUBMCDA", "SDELMA", "SDSECA", "SDUNIA",
        "PCI", "PUMAA", "BTTRA", "BTBGA",
    }
    for col in ns.columns:
        if col not in geo_id_cols:
            ns[col] = pd.to_numeric(ns[col], errors="coerce")

    ns["STATEA"] = ns["STATEA"].astype(str).str.zfill(2) if "STATEA" in ns.columns else ""
    ns["COUNTYA"] = ns["COUNTYA"].astype(str).str.zfill(3) if "COUNTYA" in ns.columns else ""
    ns["TRACTA"] = ns["TRACTA"].astype(str).str.zfill(6) if "TRACTA" in ns.columns else ""
    ns["GEOID"] = ns["STATEA"] + ns["COUNTYA"] + ns["TRACTA"]
    if "STATEA" in ns.columns and "COUNTYA" in ns.columns:
        ns["COUNTY_GEOID"] = ns["STATEA"] + ns["COUNTYA"]

    if "AU46E001" in ns.columns and "AU46E002" in ns.columns:
        total_hh = ns["AU46E001"].replace(0, np.nan)
        ns["Multigen_Rate"] = (ns["AU46E002"] / total_hh) * 100

    return ns, codebook


def build_analysis_ready_nhgis(
    data_dir: str,
    sld_filename: str = "SmartLocationDatabase.csv",
    out_path: str | None = None,
) -> pd.DataFrame:
    """
    Load NHGIS CSVs and SLD from data_dir (or data/raw/nhgis and data/raw/smart_location),
    engineer features to match ANALYSIS_READY_SCHEMA, return analysis-ready DataFrame.
    """
    # Prefer raw subfolders if present
    raw_nhgis = os.path.join(data_dir, "raw", "nhgis")
    raw_sld = os.path.join(data_dir, "raw", "smart_location")
    if os.path.isdir(raw_nhgis):
        nhgis_pattern = os.path.join(raw_nhgis, "nhgis*.csv")
    else:
        nhgis_pattern = os.path.join(data_dir, "nhgis*.csv")
    if os.path.isdir(raw_sld):
        sld_path = os.path.join(raw_sld, sld_filename)
    else:
        sld_path = os.path.join(data_dir, sld_filename)

    nhgis_files = sorted(globmod.glob(nhgis_pattern))
    if not nhgis_files:
        raise FileNotFoundError(f"No NHGIS files found: {nhgis_pattern}")

    frames = []
    for fpath in nhgis_files:
        frames.append(parse_nhgis(fpath))
    ns = frames[0]
    for extra in frames[1:]:
        new_cols = [c for c in extra.columns if c not in ns.columns]
        merge_keys = ["GISJOIN"] if "GISJOIN" in extra.columns else [ns.columns[0]]
        ns = ns.merge(extra[merge_keys + new_cols], on=merge_keys, how="outer")

    geo_id_cols = {
        "GISJOIN", "YEAR", "STUSAB", "STATE", "COUNTY", "GEO_ID", "TL_GEO_ID",
        "NAME_E", "NAME_M", "REGIONA", "DIVISIONA", "STATEA", "COUNTYA", "TRACTA",
        "BLKGRPA", "COUSUBA", "PLACEA", "CONCITA", "AIANHHA", "RES_ONLYA", "TRUSTA",
        "AIHHTLI", "AITSA", "ANRCA", "CBSAA", "CSAA", "METDIVA", "UAA", "CDCURRA",
        "SLDUA", "SLDLA", "ZCTA5A", "SUBMCDA", "SDELMA", "SDSECA", "SDUNIA",
        "PCI", "PUMAA", "BTTRA", "BTBGA",
    }
    for col in ns.columns:
        if col not in geo_id_cols:
            ns[col] = pd.to_numeric(ns[col], errors="coerce")

    ns["STATEA"] = ns["STATEA"].astype(str).str.zfill(2) if "STATEA" in ns.columns else ""
    ns["COUNTYA"] = ns["COUNTYA"].astype(str).str.zfill(3) if "COUNTYA" in ns.columns else ""
    ns["TRACTA"] = ns["TRACTA"].astype(str).str.zfill(6) if "TRACTA" in ns.columns else ""
    ns["GEOID"] = ns["STATEA"] + ns["COUNTYA"] + ns["TRACTA"]
    # County GEOID for regional fixed effects and clustered standard errors
    if "STATEA" in ns.columns and "COUNTYA" in ns.columns:
        ns["COUNTY_GEOID"] = ns["STATEA"] + ns["COUNTYA"]
    if "NAME_E" in ns.columns:
        ns["Area_Name"] = ns["NAME_E"]

    pop_total = ns["AUOVE001"].replace(0, np.nan) if "AUOVE001" in ns.columns else pd.Series(1, index=ns.index)

    # Causal target: Table B11017 household-level multigen % (AU46E002 / AU46E001 * 100)
    if "AU46E001" in ns.columns and "AU46E002" in ns.columns:
        total_hh = ns["AU46E001"].replace(0, np.nan)
        ns["Multigen_Rate"] = (ns["AU46E002"] / total_hh) * 100
        ns["_total_hh"] = ns["AU46E001"]

    if "AUOVE023" in ns.columns:
        ns["Pct_65Plus"] = (
            ns[["AUOVE020", "AUOVE021", "AUOVE022", "AUOVE023", "AUOVE024", "AUOVE025",
               "AUOVE044", "AUOVE045", "AUOVE046", "AUOVE047", "AUOVE048", "AUOVE049"]].sum(axis=1)
            / pop_total * 100
        )
        ns["Pct_Under18"] = (
            ns[["AUOVE003", "AUOVE004", "AUOVE005", "AUOVE006",
               "AUOVE027", "AUOVE028", "AUOVE029", "AUOVE030"]].sum(axis=1)
            / pop_total * 100
        )

    if "AUPFE001" in ns.columns:
        race_total = ns["AUPFE001"].replace(0, np.nan)
        ns["Pct_Hispanic"] = ns["AUPFE012"] / race_total * 100
        ns["Pct_Asian_NH"] = ns["AUPFE006"] / race_total * 100
        ns["Pct_Black_NH"] = ns["AUPFE004"] / race_total * 100
        ns["Pct_White_NH"] = ns["AUPFE003"] / race_total * 100
    if "AUYOE001" in ns.columns:
        ns["Pct_ForeignBorn"] = ns["AUYOE005"] / ns["AUYOE001"].replace(0, np.nan) * 100
    if "AURUE001" in ns.columns:
        ns["Median_HH_Income"] = ns["AURUE001"]
    if "AVA1E001" in ns.columns:
        ns["Gini_Index"] = ns["AVA1E001"]
    if "AURRE001" in ns.columns:
        ns["Poverty_Rate"] = ns["AURRE002"] / ns["AURRE001"].replace(0, np.nan) * 100
    if "AUVLE001" in ns.columns:
        st = ns["AUVLE001"].replace(0, np.nan)
        ns["Pct_SingleFamily"] = (ns["AUVLE002"] + ns["AUVLE003"]) / st * 100
        ns["Pct_5PlusUnits"] = (ns["AUVLE006"] + ns["AUVLE007"] + ns["AUVLE008"] + ns["AUVLE009"]) / st * 100
        ns["Pct_MobileHome"] = ns["AUVLE010"] / st * 100
    # Supply-side: vacancy rate (AUVUE003 + AUVUE005) / AUVUE001 * 100
    if "AUVUE001" in ns.columns:
        vu = ns["AUVUE001"].replace(0, np.nan)
        ns["Vacancy_Rate"] = (ns["AUVUE003"].fillna(0) + ns["AUVUE005"].fillna(0)) / vu * 100
    # Supply-side: % units with 4+ bedrooms (AUVRE006 + AUVRE007) / AUVRE001 * 100
    if "AUVRE001" in ns.columns:
        vr = ns["AUVRE001"].replace(0, np.nan)
        ns["Pct_LargeUnits"] = (ns["AUVRE006"].fillna(0) + ns["AUVRE007"].fillna(0)) / vr * 100
    if "AUVPE001" in ns.columns:
        ns["Median_Year_Built"] = ns["AUVPE001"]
    if "AUUEE001" in ns.columns:
        ns["Pct_Owner"] = ns["AUUEE002"] / ns["AUUEE001"].replace(0, np.nan) * 100
    if "AUUGE001" in ns.columns:
        ns["Avg_HH_Size"] = ns["AUUGE001"]
    if "AUP1E001" in ns.columns:
        ht = ns["AUP1E001"].replace(0, np.nan)
        # Feature decomposition: Non-multigen family HH = (Total Family HH - Multigen HH) / Total HH
        if "AU46E002" in ns.columns:
            ns["Pct_NonMultigen_FamilyHH"] = ((ns["AUP1E002"] - ns["AU46E002"]) / ht) * 100
        # Keep these as independent structural anchors
        ns["Pct_MarriedCouple"] = ns["AUP1E003"] / ht * 100
        ns["Pct_LivingAlone"] = ns["AUP1E008"] / ht * 100
    if "AUQNE001" in ns.columns:
        ns["Pct_HH_With_Children"] = ns["AUQNE002"] / ns["AUQNE001"].replace(0, np.nan) * 100
    if "AUQ8E001" in ns.columns:
        ed = ns["AUQ8E001"].replace(0, np.nan)
        ns["Pct_BachelorPlus"] = (ns["AUQ8E022"] + ns["AUQ8E023"] + ns["AUQ8E024"] + ns["AUQ8E025"]) / ed * 100
        less_hs_cols = [c for c in ["AUQ8E002", "AUQ8E003", "AUQ8E004", "AUQ8E005", "AUQ8E006", "AUQ8E007", "AUQ8E008", "AUQ8E009", "AUQ8E010", "AUQ8E011", "AUQ8E012", "AUQ8E013", "AUQ8E014", "AUQ8E015", "AUQ8E016"] if c in ns.columns]
        ns["Pct_LessThanHS"] = ns[less_hs_cols].sum(axis=1) / ed * 100
    if "AUPWE001" in ns.columns:
        ct = ns["AUPWE001"].replace(0, np.nan)
        ns["Pct_PublicTransit"] = ns["AUPWE010"] / ct * 100
        ns["Pct_WorkFromHome"] = ns["AUPWE021"] / ct * 100
    if "AVFIE001" in ns.columns:
        rt = ns["AVFIE001"].replace(0, np.nan)
        ns["Pct_HighRent"] = ns[["AVFIE013", "AVFIE014", "AVFIE015", "AVFIE016", "AVFIE017", "AVFIE018"]].sum(axis=1) / rt * 100
    if "AVFSE001" in ns.columns:
        ot = ns["AVFSE001"].replace(0, np.nan)
        ns["Pct_HighOwnerCost"] = ns[["AVFSE013", "AVFSE014", "AVFSE015", "AVFSE016", "AVFSE017"]].sum(axis=1) / ot * 100
    if "AU40E001" in ns.columns:
        ns["Pct_NoVehicle"] = ns["AU40E002"] / ns["AU40E001"].replace(0, np.nan) * 100
    if "AUTWE001" in ns.columns:
        ns["Pct_NotInLaborForce"] = ns["AUTWE007"] / ns["AUTWE001"].replace(0, np.nan) * 100
    if "AUTPE001" in ns.columns:
        ns["Pct_SNAP"] = ns["AUTPE002"] / ns["AUTPE001"].replace(0, np.nan) * 100
    if "AURLE001" in ns.columns:
        lt = ns["AURLE001"].replace(0, np.nan)
        ns["Pct_LimitedEnglish"] = (ns["AURLE004"] + ns["AURLE007"] + ns["AURLE010"] + ns["AURLE013"]) / lt * 100
    if "AUPHE001" in ns.columns:
        ns["Pct_SameHouse1YrAgo"] = ns["AUPHE002"] / ns["AUPHE001"].replace(0, np.nan) * 100
    if "AVH7E001" in ns.columns:
        uninsured_cols = [c for c in ns.columns if c.startswith("AVH7E") and c.endswith(("005", "008", "011", "014", "017", "020", "023", "026", "029", "032", "035", "038", "041", "044", "047", "050", "053", "056"))]
        if uninsured_cols:
            ns["Pct_Uninsured"] = ns[uninsured_cols].sum(axis=1) / ns["AVH7E001"].replace(0, np.nan) * 100

    if os.path.isfile(sld_path):
        df_sld = pd.read_csv(sld_path, dtype={"GEOID20": str, "GEOID10": str, "STATEFP": str, "COUNTYFP": str, "TRACTCE": str, "BLKGRPCE": str}, low_memory=False)
        df_sld["STATEFP"] = df_sld["STATEFP"].str.zfill(2)
        df_sld["COUNTYFP"] = df_sld["COUNTYFP"].str.zfill(3)
        df_sld["TRACTCE"] = df_sld["TRACTCE"].str.zfill(6)
        df_sld["TRACT_GEOID"] = df_sld["STATEFP"] + df_sld["COUNTYFP"] + df_sld["TRACTCE"]
        sld_agg = {k: "mean" for k in ["NatWalkInd", "D4A", "D2A_JPHH", "D3B", "D2B_E8MIXA"] if k in df_sld.columns}
        sld_tract = df_sld.groupby("TRACT_GEOID").agg(sld_agg).reset_index()
        sld_tract.rename(columns={"TRACT_GEOID": "GEOID", "D4A": "TransitFreq", "D3B": "StreetDensity", "D2B_E8MIXA": "EmpMix", "D2A_JPHH": "JobsPerHH"}, inplace=True)
        ns = ns.merge(sld_tract, on="GEOID", how="left")

    out_cols = [c for c in (ANALYSIS_READY_SCHEMA["id_cols"] + [ANALYSIS_READY_SCHEMA["target_col"], ANALYSIS_READY_SCHEMA["weight_col"]] + ANALYSIS_READY_SCHEMA["feature_cols"]) if c in ns.columns]
    out_cols = [c for c in out_cols if ns[c].notna().any()]
    result = ns[out_cols].copy()
    result = result[result[ANALYSIS_READY_SCHEMA["target_col"]].notna()].reset_index(drop=True)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        result.to_csv(out_path, index=False)
    return result
