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

# Metadata/feature columns from TIGER tract shapefile (centroids + geography)
CENTROID_COLS = ["Latitude", "Longitude"]
# Extra TIGER attributes to include in merged data for Lasso and OLS (tract geography/size)
TIGER_EXTRA_COLS = ["Tract_ALAND", "Tract_AWATER", "Tract_Shape_Area", "Tract_Shape_Leng"]

# Smart Location Database: variables to keep when joining to tract-level (GEOID) for Lasso
SLD_KEEP_COLS = [
    "CBSA", "CBSA_POP", "CBSA_EMP", "CBSA_WRK", "Ac_Total", "TotPop", "CountHU", "HH", "P_WrkAge",
    "Pct_AO0", "Pct_AO1", "Pct_AO2p", "Workers", "R_LowWageWk", "R_MedWageWk", "R_HiWageWk",
    "TotEmp", "E5_Ret", "E5_Off", "E5_Ind", "E5_Svc", "E5_Ent", "E8_Ret", "E8_off", "E8_Ind", "E8_Svc",
    "E8_Ent", "E8_Ed", "E8_Hlth", "E8_Pub", "E_LowWageWk", "E_MedWageWk", "E_HiWageWk", "E_PctLowWage",
    "D1A", "D1B", "D1C", "D1C5_RET", "D1C5_OFF", "D1C5_IND", "D1C5_SVC", "D1C5_ENT",
    "D1C8_RET", "D1C8_OFF", "D1C8_IND", "D1C8_SVC", "D1C8_ENT", "D1C8_ED", "D1C8_HLTH", "D1C8_PUB",
    "D1D", "D1_FLAG", "D2A_JPHH", "D2B_E5MIX", "D2B_E5MIXA", "D2B_E8MIX", "D2B_E8MIXA", "D2A_EPHHM",
    "D2C_TRPMX1", "D2C_TRPMX2", "D2C_TRIPEQ", "D2R_JOBPOP", "D2R_WRKEMP", "D2A_WRKEMP", "D2C_WREMLX",
    "D3A", "D3AAO", "D3AMM", "D3APO", "D3B", "D3BAO", "D3BMM3", "D3BMM4", "D3BPO3", "D3BPO4",
    "D4A", "D4B025", "D4B050", "D4C", "D4D", "D4E",
    "D5AR", "D5AE", "D5BR", "D5BE", "D5CR", "D5CRI", "D5CE", "D5CEI", "D5DR", "D5DRI", "D5DE", "D5DEI",
    "D2A_Ranked", "D2B_Ranked", "D3B_Ranked", "D4A_Ranked", "NatWalkInd",
]


def load_sld_tract(
    data_dir: str,
    sld_filename: str = "SmartLocationDatabase.csv",
) -> pd.DataFrame | None:
    """
    Load Smart Location Database CSV, aggregate to tract (GEOID = STATEFP+COUNTYFP+TRACTCE),
    return DataFrame with GEOID and only SLD_KEEP_COLS that exist. Used for Lasso and analysis-ready.
    """
    raw_sld = os.path.join(data_dir, "raw", "smart_location")
    if not os.path.isdir(raw_sld):
        raw_sld = data_dir
    sld_path = os.path.join(raw_sld, sld_filename)
    if not os.path.isfile(sld_path):
        return None
    dtype = {"GEOID10": str, "GEOID20": str, "STATEFP": str, "COUNTYFP": str, "TRACTCE": str, "BLKGRPCE": str}
    df_sld = pd.read_csv(sld_path, dtype=dtype, low_memory=False)
    if "STATEFP" in df_sld.columns and "COUNTYFP" in df_sld.columns and "TRACTCE" in df_sld.columns:
        df_sld["STATEFP"] = df_sld["STATEFP"].astype(str).str.zfill(2)
        df_sld["COUNTYFP"] = df_sld["COUNTYFP"].astype(str).str.zfill(3)
        df_sld["TRACTCE"] = df_sld["TRACTCE"].astype(str).str.zfill(6)
        df_sld["GEOID"] = df_sld["STATEFP"] + df_sld["COUNTYFP"] + df_sld["TRACTCE"]
    elif "GEOID10" in df_sld.columns:
        df_sld["GEOID10"] = df_sld["GEOID10"].astype(str).str.zfill(12)
        df_sld["GEOID"] = df_sld["GEOID10"].str[:11]
    else:
        return None
    keep = [c for c in SLD_KEEP_COLS if c in df_sld.columns]
    if not keep:
        return None
    use = ["GEOID"] + keep
    for c in keep:
        df_sld[c] = pd.to_numeric(df_sld[c], errors="coerce")
    sld_tract = df_sld[use].groupby("GEOID", as_index=False).agg("mean")
    return sld_tract


def load_tiger_centroids(data_dir: str) -> pd.DataFrame | None:
    """
    Load TIGER/Line tract attributes from NHGIS shape directories. Returns GEOID, Latitude,
    Longitude (from INTPTLAT/INTPTLON), and optional Tract_ALAND, Tract_AWATER,
    Tract_Shape_Area, Tract_Shape_Leng for Lasso/OLS. Searches unzipped tract folder first
    (nhgis0018_shapefile_tl2024_us_tract_2024), then nhgis0018_shape, nhgis0020_shape.
    Supports .dbf (on disk or inside .zip) and .csv. Returns only tract-level (11-digit GEOID).
    """
    raw_nhgis = os.path.join(data_dir, "raw", "nhgis")
    if not os.path.isdir(raw_nhgis):
        raw_nhgis = data_dir
    # Unzipped tract shapefile first, then shape dirs (zip or other)
    shape_dirs = [
        os.path.join(raw_nhgis, "nhgis0018_shapefile_tl2024_us_tract_2024"),
        os.path.join(raw_nhgis, "nhgis0018_shape"),
        os.path.join(raw_nhgis, "nhgis0020_shape"),
    ]
    shape_dirs = [d for d in shape_dirs if os.path.isdir(d)]
    if not shape_dirs:
        return None

    def _normalize_geoid(s: pd.Series) -> pd.Series:
        s = s.astype(str).str.strip()
        return s.str.zfill(11) if s.str.match(r"^\d+$").all() else s

    def _has_tract_geoid(df: pd.DataFrame) -> bool:
        if "GEOID" not in df.columns or df["GEOID"].isna().all():
            return False
        geoid_len = df["GEOID"].astype(str).str.len()
        # Tract GEOID is 11 digits (2 state + 3 county + 6 tract)
        return (geoid_len == 11).sum() >= max(1, len(df) // 2)

    def _build_geoid_from_fips(df: pd.DataFrame) -> pd.Series | None:
        for s, c, t in [("STATEFP", "COUNTYFP", "TRACTCE"), ("STATE", "COUNTY", "TRACT")]:
            if s in df.columns and c in df.columns and t in df.columns:
                geoid = (
                    df[s].astype(str).str.strip().str.zfill(2)
                    + df[c].astype(str).str.strip().str.zfill(3)
                    + df[t].astype(str).str.strip().str.zfill(6)
                )
                return geoid
        return None

    def _read_dbf_minimal(buf: bytes) -> pd.DataFrame | None:
        """Fallback: parse DBF with stdlib only (no dbfread). Returns DataFrame with string columns."""
        import struct
        if len(buf) < 32:
            return None
        num_rec = struct.unpack("<I", buf[4:8])[0]
        header_size = struct.unpack("<H", buf[8:10])[0]
        rec_size = struct.unpack("<H", buf[10:12])[0]
        fields: list[tuple[str, int, str]] = []  # name, length, type
        pos = 32
        while pos < header_size - 1 and buf[pos] != 0x0D:
            name = buf[pos : pos + 11].rstrip(b"\x00").decode("ascii", errors="ignore").strip()
            ftype = chr(buf[pos + 11]) if pos + 11 < len(buf) else "C"
            length = buf[pos + 16] if pos + 16 < len(buf) else 0
            if name:
                fields.append((name, length or 1, ftype))
            pos += 32
        if not fields:
            return None
        records: list[dict[str, str]] = []
        rec_start = header_size
        for _ in range(min(num_rec, 200000)):
            if rec_start + rec_size > len(buf):
                break
            if buf[rec_start] == 0x2A:
                rec_start += rec_size
                continue
            rec_start += 1
            row: dict[str, str] = {}
            for name, length, _ in fields:
                end = rec_start + length
                val = buf[rec_start:end].decode("utf-8", errors="replace").strip()
                row[name] = val
                rec_start = end
            records.append(row)
        if not records:
            return None
        return pd.DataFrame(records)

    def _read_dbf(path_or_bytes: str | bytes) -> pd.DataFrame | None:
        if isinstance(path_or_bytes, bytes):
            buf = path_or_bytes
        else:
            try:
                with open(path_or_bytes, "rb") as f:
                    buf = f.read()
            except OSError:
                return None
        try:
            from dbfread import DBF
        except ImportError:
            return _read_dbf_minimal(buf)
        if isinstance(path_or_bytes, bytes):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".dbf", delete=False) as f:
                f.write(path_or_bytes)
                path = f.name
            try:
                table = DBF(path, encoding="utf-8", ignore_missing_memofile=True)
                return pd.DataFrame(iter(table))
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        else:
            table = DBF(path_or_bytes, encoding="utf-8", ignore_missing_memofile=True)
            return pd.DataFrame(iter(table))

    files_to_try: list[tuple[str | bytes, str]] = []  # path or dbf bytes, ext
    for shape_dir in shape_dirs:
        for root, _dirs, files in os.walk(shape_dir):
            for f in files:
                low = f.lower()
                if low.endswith(".dbf"):
                    files_to_try.append((os.path.join(root, f), "dbf"))
                if low.endswith(".csv"):
                    files_to_try.append((os.path.join(root, f), "csv"))
                if low.endswith(".zip"):
                    # Extract .dbf from zip for tract shapefiles
                    zip_path = os.path.join(root, f)
                    try:
                        import zipfile
                        with zipfile.ZipFile(zip_path, "r") as z:
                            for name in z.namelist():
                                if name.lower().endswith(".dbf"):
                                    files_to_try.append((z.read(name), "dbf"))
                                    break
                    except Exception:
                        pass

    for path_or_buf, ext in files_to_try:
        try:
            if ext == "dbf":
                df = _read_dbf(path_or_buf)
                if df is None:
                    continue
            else:
                df = pd.read_csv(path_or_buf, dtype=str, low_memory=False, nrows=100000)
            # Prefer GEOID + INTPTLAT + INTPTLON (TIGER); else STATEFP+COUNTYFP+TRACTCE + LATITUDE+LONGITUDE (CenPop)
            lat_col = next((c for c in df.columns if c.upper() in ("INTPTLAT", "LATITUDE")), None)
            lon_col = next((c for c in df.columns if c.upper() in ("INTPTLON", "LONGITUDE")), None)
            geoid_col = next((c for c in df.columns if c.upper() == "GEOID"), None)
            if not (lat_col and lon_col):
                continue
            if geoid_col:
                geoid_ser = _normalize_geoid(df[geoid_col])
            else:
                geoid_ser = _build_geoid_from_fips(df)
                if geoid_ser is None:
                    continue
            out = pd.DataFrame({
                "GEOID": geoid_ser,
                "INTPTLAT": pd.to_numeric(df[lat_col].astype(str).str.replace("+", ""), errors="coerce"),
                "INTPTLON": pd.to_numeric(df[lon_col].astype(str).str.replace("+", ""), errors="coerce"),
            })
            # Include worthy TIGER attributes for Lasso/OLS (tract area, perimeter)
            for dbf_name, out_name in [
                ("ALAND", "Tract_ALAND"),
                ("AWATER", "Tract_AWATER"),
                ("Shape_Area", "Tract_Shape_Area"),
                ("Shape_Leng", "Tract_Shape_Leng"),
            ]:
                if dbf_name in df.columns:
                    out[out_name] = pd.to_numeric(df[dbf_name], errors="coerce")
            out = out.dropna(subset=["GEOID", "INTPTLAT", "INTPTLON"])
            if out.empty:
                continue
            if not _has_tract_geoid(out):
                continue
            return out.drop_duplicates(subset=["GEOID"], keep="first")
        except Exception:
            continue
    return None


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

    ns = ns.copy()  # Defragment after many in-place column updates to avoid PerformanceWarning

    ns["STATEA"] = ns["STATEA"].astype(str).str.zfill(2) if "STATEA" in ns.columns else ""
    ns["COUNTYA"] = ns["COUNTYA"].astype(str).str.zfill(3) if "COUNTYA" in ns.columns else ""
    ns["TRACTA"] = ns["TRACTA"].astype(str).str.zfill(6) if "TRACTA" in ns.columns else ""
    ns["GEOID"] = ns["STATEA"] + ns["COUNTYA"] + ns["TRACTA"]
    if "STATEA" in ns.columns and "COUNTYA" in ns.columns:
        ns["COUNTY_GEOID"] = ns["STATEA"] + ns["COUNTYA"]

    # TIGER tract attributes (centroids + area) for Lasso feature discovery
    tiger_df = load_tiger_centroids(data_dir)
    if tiger_df is not None:
        ns = ns.merge(tiger_df, on="GEOID", how="left")
        if "INTPTLAT" in ns.columns:
            ns = ns.rename(columns={"INTPTLAT": "Latitude", "INTPTLON": "Longitude"})
            ns["Latitude"] = pd.to_numeric(ns["Latitude"], errors="coerce")
            ns["Longitude"] = pd.to_numeric(ns["Longitude"], errors="coerce")

    sld_tract = load_sld_tract(data_dir, sld_filename="SmartLocationDatabase.csv")
    if sld_tract is not None:
        sld_cols = [c for c in sld_tract.columns if c != "GEOID"]
        ns = ns.merge(sld_tract[["GEOID"] + sld_cols], on="GEOID", how="left")
        n_sld = ns[sld_cols[0]].notna().sum() if sld_cols else 0
        print(f"SLD join: {n_sld:,} / {len(ns):,} tracts with at least one SLD variable")

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

    ns = ns.copy()  # Defragment after many in-place column updates to avoid PerformanceWarning

    ns["STATEA"] = ns["STATEA"].astype(str).str.zfill(2) if "STATEA" in ns.columns else ""
    ns["COUNTYA"] = ns["COUNTYA"].astype(str).str.zfill(3) if "COUNTYA" in ns.columns else ""
    ns["TRACTA"] = ns["TRACTA"].astype(str).str.zfill(6) if "TRACTA" in ns.columns else ""
    ns["GEOID"] = ns["STATEA"] + ns["COUNTYA"] + ns["TRACTA"]
    # County GEOID for regional fixed effects and clustered standard errors
    if "STATEA" in ns.columns and "COUNTYA" in ns.columns:
        ns["COUNTY_GEOID"] = ns["STATEA"] + ns["COUNTYA"]

    # TIGER/Line tract join: Latitude, Longitude + Tract_ALAND, etc. for spatial and Lasso
    tiger_df = load_tiger_centroids(data_dir)
    if tiger_df is None:
        print("TIGER centroid join: no tract-level file found (GEOID/INTPTLAT/INTPTLON with 11-digit GEOID in data/raw/nhgis/).")
    if tiger_df is not None:
        ns = ns.merge(tiger_df, on="GEOID", how="left")
        ns = ns.rename(columns={"INTPTLAT": "Latitude", "INTPTLON": "Longitude"})
        ns["Latitude"] = pd.to_numeric(ns["Latitude"], errors="coerce")
        ns["Longitude"] = pd.to_numeric(ns["Longitude"], errors="coerce")
        n_with_coords = ns["Latitude"].notna() & ns["Longitude"].notna()
        merge_rate = n_with_coords.sum() / len(ns) * 100 if len(ns) else 0
        print(f"TIGER centroid join: {n_with_coords.sum():,} / {len(ns):,} tracts with non-null Latitude/Longitude ({merge_rate:.2f}%)")

    if "NAME_E" in ns.columns:
        ns["Area_Name"] = ns["NAME_E"]

    pop_total = ns["AUOVE001"].replace(0, np.nan) if "AUOVE001" in ns.columns else pd.Series(1, index=ns.index)

    # Causal target: Table B11017 household-level multigen % (AU46E002 / AU46E001 * 100)
    if "AU46E001" in ns.columns and "AU46E002" in ns.columns:
        total_hh = ns["AU46E001"].replace(0, np.nan)
        ns["Multigen_Rate"] = (ns["AU46E002"] / total_hh) * 100
        ns["_total_hh"] = ns["AU46E001"]

    # Lasso discovery: Sex by Age (B01001) cell 7 — share of population
    if "AUOVE001" in ns.columns and "AUOVE007" in ns.columns:
        ns["Pct_SexByAge_Cell7"] = (ns["AUOVE007"] / ns["AUOVE001"].replace(0, np.nan)) * 100

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

    sld_tract = load_sld_tract(data_dir, sld_filename)
    if sld_tract is not None:
        sld_cols = [c for c in sld_tract.columns if c != "GEOID"]
        ns = ns.merge(sld_tract, on="GEOID", how="left")
        if "D4A" in ns.columns:
            ns["TransitFreq"] = ns["D4A"]
        if "D3B" in ns.columns:
            ns["StreetDensity"] = ns["D3B"]
        if "D2A_JPHH" in ns.columns:
            ns["JobsPerHH"] = ns["D2A_JPHH"]
        if "D2B_E8MIXA" in ns.columns:
            ns["EmpMix"] = ns["D2B_E8MIXA"]

    out_cols = [c for c in (ANALYSIS_READY_SCHEMA["id_cols"] + [ANALYSIS_READY_SCHEMA["target_col"], ANALYSIS_READY_SCHEMA["weight_col"]] + ANALYSIS_READY_SCHEMA["feature_cols"]) if c in ns.columns]
    for c in CENTROID_COLS + TIGER_EXTRA_COLS:
        if c in ns.columns and c not in out_cols:
            out_cols.append(c)
    for c in SLD_KEEP_COLS:
        if c in ns.columns and c not in out_cols:
            out_cols.append(c)
    out_cols = [c for c in out_cols if ns[c].notna().any()]
    result = ns[out_cols].copy()
    result = result[result[ANALYSIS_READY_SCHEMA["target_col"]].notna()].reset_index(drop=True)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        result.to_csv(out_path, index=False)
    return result
