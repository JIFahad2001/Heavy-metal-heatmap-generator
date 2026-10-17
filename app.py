"""
Heavy Metal Heatmap Generator — Streamlit Web Application
==========================================================
Generates publication-ready A4 concentration heatmaps (IDW interpolation)
from uploaded CSV data of geographic sampling points and heavy-metal
concentrations. Exports PNG, PDF, and ZIP.

All core functions are importable without side effects; Streamlit calls
live only inside main(), guarded by ``if __name__ == "__main__"``.
"""

from __future__ import annotations

import io
import math
import re
import string
import zipfile
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.figure as mfigure
import matplotlib.patches as mpatches
import matplotlib.patheffects as patheffects
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
import matplotlib.cm as mcm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from matplotlib.path import Path as MplPath

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree, ConvexHull, QhullError
import pyproj

# =============================================================================
# Constants (§11)
# =============================================================================
MAX_UPLOAD_MB: int = 5
MAX_ROWS: int = 5_000
MAX_METALS: int = 20
MAX_GRID_CELLS: int = 1_000_000
MAX_LABELS: int = 40
MIN_POINTS: int = 4
WARN_POINTS: int = 10
MIN_EXTENT_DEG: float = 0.002
BUFFER_FRACTION: float = 0.10

# Column aliases (§6.2)
SITE_ALIASES: list[str] = [
    "sampling sites", "sampling site", "site", "sites", "sample", "samples",
    "station", "stations", "id", "name", "sample_no", "sample no", "sample number",
    "site no", "site number", "sample_id", "sampleid", "site_id", "siteid",
    "sample_name", "location_id", "point_id",
]
LON_ALIASES: list[str] = [
    "longitudes", "longitude", "lon", "long", "lng",
]
LAT_ALIASES: list[str] = [
    "latitudes", "latitude", "lat",
]
# Columns to pre-deselect (§6.3)
NON_CONC_PATTERNS: list[str] = [
    "elevation", "depth", "ph", "ec", "id", "date", "number",
]

# Built-in element names for title rendering (§9.2)
ELEMENT_NAMES: dict[str, str] = {
    "Pb": "Lead", "Cd": "Cadmium", "Cr": "Chromium", "Cu": "Copper",
    "Zn": "Zinc", "Ni": "Nickel", "Hg": "Mercury", "As": "Arsenic",
    "Fe": "Iron", "Mn": "Manganese", "Co": "Cobalt", "Se": "Selenium",
    "Mo": "Molybdenum", "V": "Vanadium", "Ba": "Barium", "Sb": "Antimony",
    "Sn": "Tin", "Ti": "Titanium", "Tl": "Thallium", "Al": "Aluminium",
}

# 9-Class GIS Yellow-to-Dark-Brown colormap (matching reference GIS publications)
GIS_COLORS_9: list[str] = [
    '#ffff80',  # Class 1: light yellow
    '#fce96a',  # Class 2: warm yellow
    '#f7d154',  # Class 3: golden yellow
    '#f0b641',  # Class 4: yellow-orange
    '#e39832',  # Class 5: orange-ochre
    '#d17824',  # Class 6: ochre-brown
    '#b85818',  # Class 7: medium-dark brown
    '#94370e',  # Class 8: dark reddish-brown
    '#6b1404',  # Class 9: very dark maroon-brown
]
GIS_CMAP_9 = mcolors.ListedColormap(GIS_COLORS_9, name="gis_yellow_brown")
if "gis_yellow_brown" not in matplotlib.colormaps:
    try:
        matplotlib.colormaps.register(GIS_CMAP_9, name="gis_yellow_brown")
    except Exception:
        pass

# Sequential and diverging colour ramps (§8)
SEQUENTIAL_CMAPS: list[str] = ["gis_yellow_brown", "viridis", "plasma", "inferno", "magma", "cividis"]
DIVERGING_CMAPS: list[str] = ["coolwarm", "RdYlBu", "Spectral"]
ALL_CMAPS: list[str] = SEQUENTIAL_CMAPS + DIVERGING_CMAPS

# Non-detect patterns (§6.4)
NON_DETECT_RE = re.compile(r"^\s*(<\s*[\d.]+|nd|bdl|<\s*dl)\s*$", re.IGNORECASE)

# CSV template
CSV_TEMPLATE = (
    "Sampling Sites,Longitudes,Latitudes,Pb,Cd,Cr,Cu\n"
    "Site 1,90.412,23.810,12.5,0.8,25.4,18.2\n"
    "Site 2,90.425,23.796,15.2,1.1,21.7,16.9\n"
    "Site 3,90.438,23.785,11.8,0.6,28.1,20.5\n"
    "Site 4,90.450,23.800,14.1,0.9,23.6,19.0\n"
    "Site 5,90.420,23.815,13.3,0.7,26.8,17.5\n"
)


# =============================================================================
# §6 — Data validation
# =============================================================================


def read_csv_flexible(file_bytes: bytes) -> pd.DataFrame:
    """Read a CSV from raw bytes, trying UTF-8-BOM then Windows-1252,
    auto-detecting delimiter among comma, semicolon, tab.
    For semicolon files, treat commas in numeric fields as decimal separators."""
    text: str | None = None
    for enc in ("utf-8-sig", "cp1252"):
        try:
            text = file_bytes.decode(enc)
            break
        except (UnicodeDecodeError, ValueError):
            continue
    if text is None:
        raise ValueError("Cannot decode the file as UTF-8 or Windows-1252.")
    if not text.strip():
        raise ValueError("The uploaded file is empty.")

    # Detect delimiter from the first line
    first_line = text.split("\n", 1)[0]
    if "\t" in first_line:
        sep = "\t"
    elif ";" in first_line:
        sep = ";"
    else:
        sep = ","

    decimal = "."
    if sep == ";":
        # Semicolon-separated files often use comma as decimal
        decimal = ","

    df = pd.read_csv(
        io.StringIO(text), sep=sep, decimal=decimal,
        engine="python", on_bad_lines="skip",
    )
    return df


def _match_alias(col: str, aliases: list[str]) -> bool:
    """Case-insensitive match of a trimmed column name against aliases."""
    c = col.strip().lower()
    c_clean = re.sub(r"[-_.\s]+", " ", c).strip()
    return c in aliases or c_clean in aliases


def _is_site_column(col: str) -> bool:
    """Return True if col represents sample or site identifiers."""
    c = col.strip().lower()
    c_clean = re.sub(r"[-_.\s]+", " ", c).strip()
    if c in SITE_ALIASES or c_clean in SITE_ALIASES:
        return True
    # Any column name containing sample, site, station, or sampling
    for kw in ["sample", "sampling", "site", "station"]:
        if kw in c_clean:
            return True
    tokens = c_clean.split()
    if any(t in {"sample", "samples", "site", "sites", "station", "stations"} for t in tokens):
        return True
    return False


def _find_site_column(df: pd.DataFrame) -> str | None:
    """Find the best column representing sampling sites or sample numbers."""
    for col in df.columns:
        if _is_site_column(col):
            return col
    for col in df.columns:
        if _match_alias(col, ["id", "name", "point", "loc", "location"]):
            return col
    return None


def _find_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    """Return the first column in *df* whose trimmed-lowercase name matches
    any alias.  Returns ``None`` if nothing matches."""
    for col in df.columns:
        if _match_alias(col, aliases):
            return col
    return None


def _parse_header_unit(header: str) -> tuple[str, str]:
    """Split e.g. 'Pb (mg/kg)' → ('Pb', 'mg/kg').
    If no parenthesised unit, returns (header_stripped, '')."""
    m = re.match(r"^(.+?)\s*\(([^)]+)\)\s*$", header.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return header.strip(), ""


def detect_columns(
    df: pd.DataFrame,
) -> tuple[str | None, str | None, str | None, list[str]]:
    """Detect site, lon, lat columns and return (site_col, lon_col, lat_col,
    list_of_all_col_names).  Returns None for unmatched required columns."""
    site_col = _find_site_column(df)
    lon_col = _find_column(df, LON_ALIASES)
    lat_col = _find_column(df, LAT_ALIASES)
    return site_col, lon_col, lat_col, list(df.columns)


def _is_non_concentration(name: str) -> bool:
    """Return True if *name* looks like a non-concentration column."""
    low = name.strip().lower()
    # Remove parenthesised units first
    low = re.sub(r"\s*\([^)]*\)", "", low).strip()
    for pat in NON_CONC_PATTERNS:
        if pat in low:
            return True
    return False


def detect_metals(
    df: pd.DataFrame,
    site_col: str | None,
    lon_col: str,
    lat_col: str,
) -> list[dict[str, Any]]:
    """Return a list of candidate metal columns.

    Each entry is a dict:
        name: original column header
        symbol: extracted symbol (e.g. 'Pb')
        unit: extracted unit (e.g. 'mg/kg') or ''
        valid_count: how many cells convert to a number
        total_non_empty: non-empty cells
        pre_selected: True unless it looks like a non-concentration
        usable: True if at least one valid numeric value
    """
    skip = {lon_col, lat_col}
    if site_col is not None:
        skip.add(site_col)

    candidates = []
    for col in df.columns:
        if col in skip:
            continue
        series = df[col]
        non_empty = series.dropna()
        non_empty = non_empty[non_empty.astype(str).str.strip() != ""]
        total_non_empty = len(non_empty)
        if total_non_empty == 0:
            continue
        # Try numeric conversion
        numeric = pd.to_numeric(non_empty, errors="coerce")
        valid_count = int(numeric.notna().sum())
        if valid_count < 0.5 * total_non_empty:
            continue
        symbol, unit = _parse_header_unit(col)
        candidates.append({
            "name": col,
            "symbol": symbol,
            "unit": unit,
            "valid_count": valid_count,
            "total_non_empty": total_non_empty,
            "pre_selected": not _is_non_concentration(col),
            "usable": valid_count > 0,
        })
    return candidates


def _classify_value(raw: Any) -> tuple[str, float | None]:
    """Classify a single cell value.

    Returns (category, numeric_value_or_None).
    Categories: 'valid', 'zero', 'non_detect', 'negative', 'invalid', 'missing'
    """
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return "missing", None
    s = str(raw).strip()
    if s == "":
        return "missing", None
    if NON_DETECT_RE.match(s):
        return "non_detect", None
    try:
        v = float(s)
    except (ValueError, TypeError):
        return "invalid", None
    if math.isinf(v) or math.isnan(v):
        return "invalid", None
    if v < 0:
        return "negative", None
    if v == 0:
        return "zero", 0.0
    return "valid", v


def process_metal_values(
    series: pd.Series,
) -> tuple[np.ndarray, dict[str, int]]:
    """Convert a column of raw values to a float array with NaN for unmappable cells.

    Returns (values_array, counts_dict).
    counts_dict has keys: valid, zero, non_detect, negative, invalid, missing.
    """
    counts: dict[str, int] = {
        "valid": 0, "zero": 0, "non_detect": 0,
        "negative": 0, "invalid": 0, "missing": 0,
    }
    result = np.full(len(series), np.nan)
    for i, raw in enumerate(series):
        cat, val = _classify_value(raw)
        counts[cat] += 1
        if cat == "valid":
            result[i] = val
        elif cat == "zero":
            result[i] = 0.0
    return result, counts


def validate_coordinates(
    lon_series: pd.Series,
    lat_series: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Validate longitude and latitude columns.

    Returns (lon_array, lat_array, valid_mask, report_dict).
    valid_mask is True for rows with valid finite coordinates.
    """
    n = len(lon_series)
    lon_arr = pd.to_numeric(lon_series, errors="coerce").values.astype(float)
    lat_arr = pd.to_numeric(lat_series, errors="coerce").values.astype(float)

    valid = np.ones(n, dtype=bool)
    reasons: dict[str, int] = {
        "non_numeric": 0,
        "out_of_range": 0,
        "origin": 0,
    }

    for i in range(n):
        lon, lat = lon_arr[i], lat_arr[i]
        if not np.isfinite(lon) or not np.isfinite(lat):
            valid[i] = False
            reasons["non_numeric"] += 1
            continue
        if lon < -180 or lon > 180 or lat < -90 or lat > 90:
            valid[i] = False
            reasons["out_of_range"] += 1
            continue
        if lon == 0.0 and lat == 0.0:
            reasons["origin"] += 1
            # Keep but flag

    # Check for DMS text — if any original values contain ° or ' or " symbols
    dms_detected = False
    for s in list(lon_series.astype(str)) + list(lat_series.astype(str)):
        if any(c in str(s) for c in ["°", "'", '"', "′", "″"]):
            dms_detected = True
            break

    # Check if coordinates might be swapped
    swap_suggested = False
    finite_mask = np.isfinite(lon_arr) & np.isfinite(lat_arr)
    if np.any(finite_mask):
        finite_lats = lat_arr[finite_mask]
        finite_lons = lon_arr[finite_mask]
        if np.all(np.abs(finite_lats) > 90) and np.all(np.abs(finite_lons) <= 90):
            swap_suggested = True

    report = {
        "total": n,
        "valid": int(valid.sum()),
        "excluded": int((~valid).sum()),
        "reasons": reasons,
        "dms_detected": dms_detected,
        "swap_suggested": swap_suggested,
        "has_origin": reasons["origin"] > 0,
    }

    return lon_arr, lat_arr, valid, report


def average_duplicates(
    lons: np.ndarray,
    lats: np.ndarray,
    values: np.ndarray,
    decimals: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Average points with identical coordinates (rounded).

    Returns (unique_lons, unique_lats, averaged_values, n_duplicate_groups).
    Only considers rows where values is not NaN.
    """
    mask = np.isfinite(values)
    if mask.sum() == 0:
        return np.array([]), np.array([]), np.array([]), 0

    lons_v = lons[mask]
    lats_v = lats[mask]
    vals_v = values[mask]

    rounded_lon = np.round(lons_v, decimals)
    rounded_lat = np.round(lats_v, decimals)

    # Group by rounded coordinates
    coord_keys = {}
    for i in range(len(lons_v)):
        key = (rounded_lon[i], rounded_lat[i])
        if key not in coord_keys:
            coord_keys[key] = []
        coord_keys[key].append(vals_v[i])

    n_groups = sum(1 for v in coord_keys.values() if len(v) > 1)
    u_lons = np.array([k[0] for k in coord_keys])
    u_lats = np.array([k[1] for k in coord_keys])
    u_vals = np.array([np.mean(v) for v in coord_keys.values()])

    return u_lons, u_lats, u_vals, n_groups


# =============================================================================
# §7 — Extent, CRS, IDW, masking
# =============================================================================


def compute_extent(
    all_lons: np.ndarray,
    all_lats: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compute the buffered extent from all retained locations.

    Returns (lon_min, lon_max, lat_min, lat_max) with 10% buffer.
    Handles degenerate (near-zero range) axes per §7.1.
    """
    lon_min_raw = float(np.min(all_lons))
    lon_max_raw = float(np.max(all_lons))
    lat_min_raw = float(np.min(all_lats))
    lat_max_raw = float(np.max(all_lats))

    d_lon = lon_max_raw - lon_min_raw
    d_lat = lat_max_raw - lat_min_raw

    # Handle degenerate axes
    if d_lon < MIN_EXTENT_DEG:
        replacement = max(MIN_EXTENT_DEG, 0.25 * d_lat)
        mid_lon = (lon_min_raw + lon_max_raw) / 2.0
        lon_min_raw = mid_lon - replacement / 2.0
        lon_max_raw = mid_lon + replacement / 2.0
        d_lon = replacement

    if d_lat < MIN_EXTENT_DEG:
        replacement = max(MIN_EXTENT_DEG, 0.25 * d_lon)
        mid_lat = (lat_min_raw + lat_max_raw) / 2.0
        lat_min_raw = mid_lat - replacement / 2.0
        lat_max_raw = mid_lat + replacement / 2.0
        d_lat = replacement

    lon_min = lon_min_raw - BUFFER_FRACTION * d_lon
    lon_max = lon_max_raw + BUFFER_FRACTION * d_lon
    lat_min = lat_min_raw - BUFFER_FRACTION * d_lat
    lat_max = lat_max_raw + BUFFER_FRACTION * d_lat

    return lon_min, lon_max, lat_min, lat_max


def determine_crs(
    centroid_lon: float,
    centroid_lat: float,
    lon_span: float,
) -> tuple[pyproj.CRS, str, bool]:
    """Choose CRS for interpolation: UTM or azimuthal equidistant.

    Returns (crs_object, epsg_or_proj_string, is_approximate).
    """
    approximate = False
    if lon_span > 6.0 or abs(centroid_lat) > 80.0:
        # Azimuthal equidistant
        proj_str = (
            f"+proj=aeqd +lat_0={centroid_lat} +lon_0={centroid_lon} "
            f"+datum=WGS84 +units=m +no_defs"
        )
        crs = pyproj.CRS.from_proj4(proj_str)
        approximate = True
        return crs, proj_str, approximate

    # UTM zone
    zone = int((centroid_lon + 180) / 6) + 1
    zone = max(1, min(60, zone))
    if centroid_lat >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone
    crs = pyproj.CRS.from_epsg(epsg)
    return crs, f"EPSG:{epsg}", False


def project_points(
    lons: np.ndarray,
    lats: np.ndarray,
    target_crs: pyproj.CRS,
) -> tuple[np.ndarray, np.ndarray]:
    """Project lon/lat (WGS84) to the target CRS. Returns (x, y)."""
    wgs84 = pyproj.CRS.from_epsg(4326)
    transformer = pyproj.Transformer.from_crs(
        wgs84, target_crs, always_xy=True,
    )
    x, y = transformer.transform(lons, lats)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def project_extent(
    lon_min: float, lon_max: float,
    lat_min: float, lat_max: float,
    target_crs: pyproj.CRS,
) -> tuple[float, float, float, float]:
    """Project the buffered extent box (corners + edge midpoints) and
    return the projected bounding box (x_min, x_max, y_min, y_max)."""
    corners_lon = [lon_min, lon_max, lon_min, lon_max,
                   (lon_min + lon_max) / 2, (lon_min + lon_max) / 2,
                   lon_min, lon_max]
    corners_lat = [lat_min, lat_min, lat_max, lat_max,
                   lat_min, lat_max,
                   (lat_min + lat_max) / 2, (lat_min + lat_max) / 2]
    x, y = project_points(
        np.array(corners_lon), np.array(corners_lat), target_crs,
    )
    return float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))


def build_grid(
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Build the interpolation grid.

    *resolution* cells along the longer side, capped at MAX_GRID_CELLS.
    Returns (GX, GY, nx, ny).
    """
    dx = x_max - x_min
    dy = y_max - y_min
    if dx >= dy:
        nx = resolution
        ny = max(1, int(round(resolution * dy / dx)))
    else:
        ny = resolution
        nx = max(1, int(round(resolution * dx / dy)))
    # Cap total cells
    total = nx * ny
    if total > MAX_GRID_CELLS:
        scale = math.sqrt(MAX_GRID_CELLS / total)
        nx = max(1, int(nx * scale))
        ny = max(1, int(ny * scale))
    gx = np.linspace(x_min, x_max, nx)
    gy = np.linspace(y_min, y_max, ny)
    GX, GY = np.meshgrid(gx, gy)
    return GX, GY, nx, ny


def idw_interpolate(
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    sample_z: np.ndarray,
    GX: np.ndarray,
    GY: np.ndarray,
    power: float = 2.0,
    k: int | None = None,
    chunk_size: int = 10_000,
) -> np.ndarray:
    """IDW interpolation using cKDTree.

    Parameters
    ----------
    sample_x, sample_y : arrays of sample coordinates (projected metres).
    sample_z : array of concentration values.
    GX, GY : meshgrid arrays for the interpolation grid.
    power : IDW exponent.
    k : number of nearest neighbours (default min(n, 24)).
    chunk_size : cells per processing chunk.

    Returns
    -------
    Z : 2D array of interpolated values, same shape as GX.
    """
    n = len(sample_x)
    if k is None:
        k = min(n, 24)
    k = min(k, n)

    points = np.column_stack([sample_x, sample_y])
    tree = cKDTree(points)

    grid_flat = np.column_stack([GX.ravel(), GY.ravel()])
    Z_flat = np.empty(grid_flat.shape[0])

    for start in range(0, len(grid_flat), chunk_size):
        end = min(start + chunk_size, len(grid_flat))
        chunk = grid_flat[start:end]
        dists, indices = tree.query(chunk, k=k)
        if k == 1:
            dists = dists.reshape(-1, 1)
            indices = indices.reshape(-1, 1)

        # Handle coincident points
        coincident = dists[:, 0] < 1e-10
        dists = np.maximum(dists, 1e-10)
        weights = 1.0 / (dists ** power)
        z_vals = sample_z[indices]
        weighted_sum = np.sum(weights * z_vals, axis=1)
        weight_total = np.sum(weights, axis=1)
        result = weighted_sum / weight_total

        # For coincident points, use exact sample value
        if np.any(coincident):
            result[coincident] = sample_z[indices[coincident, 0]]

        Z_flat[start:end] = result

    return Z_flat.reshape(GX.shape)


def compute_convex_hull_mask(
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    GX: np.ndarray,
    GY: np.ndarray,
) -> np.ndarray | None:
    """Compute convex hull of sample points and return a boolean mask
    (True = inside hull) for the grid.  Returns None if points are collinear."""
    if len(sample_x) < 3:
        return None
    try:
        hull = ConvexHull(np.column_stack([sample_x, sample_y]))
    except QhullError:
        return None  # Collinear
    hull_points = np.column_stack([sample_x, sample_y])[hull.vertices]
    # Close the polygon
    hull_path = MplPath(np.vstack([hull_points, hull_points[0]]))
    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])
    inside = hull_path.contains_points(grid_pts).reshape(GX.shape)
    return inside


def compute_distance_mask(
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    GX: np.ndarray,
    GY: np.ndarray,
    multiplier: float = 3.0,
) -> np.ndarray:
    """Mask cells beyond multiplier × median nearest-neighbour distance."""
    if len(sample_x) < 2:
        # Can't compute NN distance with 1 point
        return np.ones(GX.shape, dtype=bool)
    points = np.column_stack([sample_x, sample_y])
    tree = cKDTree(points)
    nn_dists, _ = tree.query(points, k=2)
    median_nn = float(np.median(nn_dists[:, 1]))
    max_dist = multiplier * median_nn

    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])
    min_dists, _ = tree.query(grid_pts, k=1)
    within = (min_dists <= max_dist).reshape(GX.shape)
    return within


def apply_mask(
    Z: np.ndarray,
    GX: np.ndarray,
    GY: np.ndarray,
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    full_extent: bool = False,
    distance_multiplier: float = 3.0,
) -> np.ndarray:
    """Apply convex-hull and distance masks to the interpolated grid.

    Returns a masked array. If full_extent is True, no masking is applied.
    """
    if full_extent:
        return np.ma.array(Z, mask=False)

    hull_mask = compute_convex_hull_mask(sample_x, sample_y, GX, GY)
    dist_mask = compute_distance_mask(
        sample_x, sample_y, GX, GY, distance_multiplier,
    )

    combined = dist_mask.copy()
    if hull_mask is not None:
        combined = combined & hull_mask

    return np.ma.array(Z, mask=~combined)


def leave_one_out_cv(
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    sample_z: np.ndarray,
    power: float = 2.0,
) -> tuple[float, float]:
    """Leave-one-out cross-validation for IDW.

    Returns (RMSE, MAE).
    """
    n = len(sample_x)
    if n < 2:
        return float("nan"), float("nan")
    errors = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        other_x = sample_x[mask]
        other_y = sample_y[mask]
        other_z = sample_z[mask]

        k = min(len(other_x), 24)
        tree = cKDTree(np.column_stack([other_x, other_y]))
        dists, indices = tree.query([[sample_x[i], sample_y[i]]], k=k)
        if k == 1:
            dists = dists.reshape(1, 1)
            indices = indices.reshape(1, 1)
        dists = np.maximum(dists, 1e-10)
        weights = 1.0 / (dists ** power)
        predicted = np.sum(weights * other_z[indices], axis=1) / np.sum(weights, axis=1)
        errors[i] = sample_z[i] - predicted[0]

    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))
    return rmse, mae


# =============================================================================
# §9 — Rendering / A4 layout
# =============================================================================

def _compute_scale_bar(x_range_m: float) -> tuple[float, str]:
    """Compute a nice scale bar length. Returns (length_m, label)."""
    target = x_range_m * 0.2  # about 20% of map width
    nice_vals = []
    for base in [1, 2, 5]:
        for exp in range(-1, 8):
            nice_vals.append(base * (10 ** exp))
    nice_vals.sort()
    best = nice_vals[0]
    for nv in nice_vals:
        if nv <= target:
            best = nv
        else:
            break
    if best >= 1000:
        label = f"{best / 1000:g} km"
    else:
        label = f"{best:g} m"
    return float(best), label


def _draw_north_arrow(ax: Any, x: float, y: float, size: float = 0.05) -> None:
    """Draw a north arrow at axes-fraction coordinates (x, y)."""
    trans = ax.transAxes
    # Simple north arrow: triangle + N label
    arrow_h = size
    arrow_w = size * 0.4

    # Main triangle
    triangle = mpatches.FancyArrow(
        x, y - arrow_h / 2, 0, arrow_h,
        head_width=arrow_w, head_length=arrow_h * 0.4,
        fc="black", ec="black", linewidth=0.5,
        transform=trans, zorder=20,
    )
    ax.add_patch(triangle)
    ax.text(
        x, y + arrow_h / 2 + 0.015, "N",
        transform=trans, ha="center", va="bottom",
        fontsize=9, fontweight="bold", family="DejaVu Sans",
        zorder=20,
    )


def format_dms_lon(val: float) -> str:
    sign = "E" if val >= 0 else "W"
    v = abs(val)
    deg = int(v)
    rem = (v - deg) * 60
    mins = int(rem)
    secs = int(round((rem - mins) * 60))
    if secs == 60:
        mins += 1
        secs = 0
    return f"{deg}°{mins:02d}'{secs:02d}\"{sign}"


def format_dms_lat(val: float) -> str:
    sign = "N" if val >= 0 else "S"
    v = abs(val)
    deg = int(v)
    rem = (v - deg) * 60
    mins = int(rem)
    secs = int(round((rem - mins) * 60))
    if secs == 60:
        mins += 1
        secs = 0
    return f"{deg}°{mins:02d}'{secs:02d}\"{sign}"


def _draw_gis_compass_rose(ax: Any, x: float, y: float, size: float = 0.075, aspect_ratio: float = 1.0) -> None:
    """Draw publication 8-point compass rose in Axes fraction coordinates."""
    trans = ax.transAxes
    sx = size / max(aspect_ratio, 0.01)
    sy = size

    c1 = mpatches.Ellipse((x, y), sx * 1.16, sy * 1.16, facecolor="none", edgecolor="black", linewidth=0.8, transform=trans, zorder=15)
    c2 = mpatches.Ellipse((x, y), sx * 0.96, sy * 0.96, facecolor="none", edgecolor="black", linewidth=0.5, transform=trans, zorder=15)
    ax.add_patch(c1)
    ax.add_patch(c2)

    r_main = 0.72
    r_diag = 0.45
    r_in = 0.14

    for ang in [90, 270, 0, 180]:
        rad = np.deg2rad(ang)
        tip = (x + sx * r_main * np.cos(rad), y + sy * r_main * np.sin(rad))
        left = (x + sx * r_in * np.cos(np.deg2rad(ang + 90)), y + sy * r_in * np.sin(np.deg2rad(ang + 90)))
        right = (x + sx * r_in * np.cos(np.deg2rad(ang - 90)), y + sy * r_in * np.sin(np.deg2rad(ang - 90)))
        ax.add_patch(mpatches.Polygon([(x, y), tip, left], closed=True, facecolor="black", edgecolor="black", linewidth=0.5, transform=trans, zorder=16))
        ax.add_patch(mpatches.Polygon([(x, y), tip, right], closed=True, facecolor="white", edgecolor="black", linewidth=0.5, transform=trans, zorder=16))

    for ang in [45, 135, 225, 315]:
        rad = np.deg2rad(ang)
        tip = (x + sx * r_diag * np.cos(rad), y + sy * r_diag * np.sin(rad))
        left = (x + sx * r_in * 0.6 * np.cos(np.deg2rad(ang + 90)), y + sy * r_in * 0.6 * np.sin(np.deg2rad(ang + 90)))
        right = (x + sx * r_in * 0.6 * np.cos(np.deg2rad(ang - 90)), y + sy * r_in * 0.6 * np.sin(np.deg2rad(ang - 90)))
        ax.add_patch(mpatches.Polygon([left, tip, right], closed=True, facecolor="white", edgecolor="black", linewidth=0.4, transform=trans, zorder=15))

    ax.text(x, y + sy * r_main + 0.015, "N", transform=trans, ha="center", va="bottom", fontsize=8.0, fontweight="bold", family="serif", zorder=17)
    ax.text(x, y - sy * r_main - 0.015, "S", transform=trans, ha="center", va="top", fontsize=7.2, fontweight="bold", family="serif", zorder=17)
    ax.text(x + sx * r_main + 0.010, y, "E", transform=trans, ha="left", va="center", fontsize=7.2, fontweight="bold", family="serif", zorder=17)
    ax.text(x - sx * r_main - 0.010, y, "W", transform=trans, ha="right", va="center", fontsize=7.2, fontweight="bold", family="serif", zorder=17)


def _draw_gis_bracket_scale_bar(ax: Any, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
    """Draw publication bracket scale bar in data coordinates."""
    x_span = x_max - x_min
    y_span = y_max - y_min
    target_len = x_span * 0.22
    nice_steps = [100, 200, 500, 1000, 2000, 4000, 5000, 10000, 20000, 50000, 100000]
    best_len = nice_steps[0]
    for step in nice_steps:
        if step <= target_len:
            best_len = step
        else:
            break

    total_val = best_len
    if total_val >= 1000:
        unit_str = "Km"
        vals = [0, total_val * 0.125, total_val * 0.25, total_val * 0.5, total_val * 0.75, total_val]
        labels = [f"{v/1000:g}" for v in vals[:-1]] + [f"{total_val/1000:g} {unit_str}"]
    else:
        unit_str = "m"
        vals = [0, total_val * 0.25, total_val * 0.5, total_val]
        labels = [f"{v:g}" for v in vals[:-1]] + [f"{total_val:g} {unit_str}"]

    x0 = x_min + 0.045 * x_span
    y0 = y_min + 0.060 * y_span
    h = 0.016 * y_span

    for i in range(len(vals) - 1):
        xa = x0 + (vals[i] / total_val) * total_val
        xb = x0 + (vals[i + 1] / total_val) * total_val
        y_top = y0 + h if i % 2 == 0 else y0 + 0.55 * h
        ax.plot([xa, xb], [y_top, y_top], color="black", lw=0.85, zorder=15)
        ax.plot([xa, xa], [y0, y_top], color="black", lw=0.85, zorder=15)
        ax.plot([xb, xb], [y0, y_top], color="black", lw=0.85, zorder=15)
    ax.plot([x0, x0 + total_val], [y0, y0], color="black", lw=0.85, zorder=15)

    for idx, (tk, lbl) in enumerate(zip(vals, labels)):
        xt = x0 + tk
        ax.text(xt, y0 - 0.010 * y_span, lbl, ha="center", va="top", fontsize=7.2, family="sans-serif", zorder=15)


def _inverse_project_grid(
    GX: np.ndarray, GY: np.ndarray, target_crs: pyproj.CRS,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-project grid coordinates back to lon/lat for graticule."""
    wgs84 = pyproj.CRS.from_epsg(4326)
    transformer = pyproj.Transformer.from_crs(
        target_crs, wgs84, always_xy=True,
    )
    lons, lats = transformer.transform(GX, GY)
    return lons, lats


def plot_metal_map_axes(
    ax: matplotlib.axes.Axes,
    fig: mfigure.Figure,
    metal_symbol: str,
    metal_unit: str,
    Z_masked: np.ma.MaskedArray,
    GX: np.ndarray,
    GY: np.ndarray,
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    sample_z: np.ndarray,
    sample_labels: np.ndarray | None,
    proj_extent: tuple[float, float, float, float],
    target_crs: pyproj.CRS,
    crs_label: str,
    settings: dict[str, Any],
    rmse: float,
    mae: float,
    n_points: int,
    panel_label: str | None = None,
    is_subpanel: bool = False,
) -> None:
    """Plot an individual metal concentration heatmap onto a given Axes."""
    cmap_name = settings.get("cmap", "gis_yellow_brown")
    if cmap_name == "gis_yellow_brown":
        cmap = GIS_CMAP_9.copy()
    else:
        try:
            cmap = matplotlib.colormaps[cmap_name].copy()
        except KeyError:
            cmap = matplotlib.colormaps["viridis"].copy()
    if settings.get("reverse_cmap", False):
        cmap = cmap.reversed()
    if hasattr(cmap, "with_extremes"):
        cmap = cmap.with_extremes(bad=(0, 0, 0, 0))
    else:
        cmap.set_bad(alpha=0)

    # Colour limits
    valid_data = Z_masked.compressed()
    if len(valid_data) == 0:
        vmin, vmax = 0, 1
    else:
        vmin, vmax = float(np.min(sample_z)), float(np.max(sample_z))

    constant_value = False
    colour_scale = settings.get("colour_scale", "linear")
    extend = "neither"

    if vmin == vmax:
        constant_value = True
        if vmin == 0:
            vmin, vmax = -1, 1
        else:
            vmin = vmin - abs(vmin) * 0.1
            vmax = vmax + abs(vmax) * 0.1

    if colour_scale == "log":
        if np.all(sample_z > 0):
            norm = mcolors.LogNorm(vmin=max(vmin, 1e-10), vmax=vmax)
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    elif colour_scale == "percentile clip":
        p2 = float(np.percentile(sample_z, 2))
        p98 = float(np.percentile(sample_z, 98))
        if p2 < p98:
            vmin, vmax = p2, p98
            extend = "both"
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    x_min, x_max, y_min, y_max = proj_extent
    opacity = float(settings.get("opacity", 1.0))

    map_style = settings.get("map_style", "GIS Journal / Reference Style (Publication)")
    is_gis_style = (
        map_style == "GIS Journal / Reference Style (Publication)"
        or settings.get("style") == "gis_publication"
        or cmap_name == "gis_yellow_brown"
    )

    if is_gis_style:
        ax.set_facecolor("white")

        # Data bounding box in projected metres
        data_x_min = float(np.min(sample_x))
        data_x_max = float(np.max(sample_x))
        data_y_min = float(np.min(sample_y))
        data_y_max = float(np.max(sample_y))

        dx = max(data_x_max - data_x_min, 1000.0)
        dy = max(data_y_max - data_y_min, 1000.0)

        gis_x_min = data_x_min - 0.12 * dx
        gis_x_max = data_x_max + 0.65 * dx
        gis_y_min = data_y_min - 0.28 * dy
        gis_y_max = data_y_max + 0.28 * dy

        ax.set_xlim(gis_x_min, gis_x_max)
        ax.set_ylim(gis_y_min, gis_y_max)
        ax.set_aspect("equal")

        x_span = gis_x_max - gis_x_min
        y_span = gis_y_max - gis_y_min
        box_aspect = x_span / max(y_span, 1.0)

        # 9 discrete intervals
        bounds = np.linspace(vmin, vmax, 10)
        norm_b = BoundaryNorm(bounds, cmap.N, clip=True)

        try:
            im = ax.contourf(
                GX, GY, Z_masked,
                levels=bounds, cmap=cmap, norm=norm_b, alpha=opacity, zorder=2,
            )
            ax.contour(
                GX, GY, Z_masked,
                levels=bounds[1:-1], colors="#8c5627", linewidths=0.35, alpha=0.45, zorder=3,
            )
        except Exception:
            im = ax.pcolormesh(
                GX, GY, Z_masked,
                cmap=cmap, norm=norm_b, alpha=opacity, shading="auto", zorder=2,
            )

        # Buffered hull outline in grey
        if len(sample_x) >= 3:
            try:
                hull = ConvexHull(np.column_stack([sample_x, sample_y]))
                hull_pts = np.column_stack([sample_x, sample_y])[hull.vertices]
                hull_pts = np.vstack([hull_pts, hull_pts[0]])
                centroid = np.mean(hull_pts[:-1], axis=0)
                buffered_hull = centroid + 1.10 * (hull_pts - centroid)
                ax.plot(buffered_hull[:, 0], buffered_hull[:, 1], color="#6e6e6e", linewidth=0.85, zorder=4)
            except Exception:
                pass

        # Sampling markers: dark crimson
        show_points = settings.get("show_points", True)
        point_size = settings.get("point_size", 30)
        if show_points:
            ax.scatter(
                sample_x, sample_y,
                s=point_size if not is_subpanel else max(16, point_size * 0.75),
                c="#7a0118", edgecolors="#2b0008",
                linewidths=0.5, zorder=6,
            )

        # Site labels
        show_labels = settings.get("show_labels", False)
        if show_labels and sample_labels is not None:
            n_labels = min(len(sample_labels), len(sample_x))
            if n_labels <= MAX_LABELS:
                for i in range(n_labels):
                    lbl = str(sample_labels[i])[:18].replace("$", r"\$")
                    txt = ax.annotate(
                        lbl,
                        (sample_x[i], sample_y[i]),
                        xytext=(4, 3), textcoords="offset points",
                        fontsize=6.8 if is_subpanel else 7.5,
                        color="#1a1a1a", family="DejaVu Sans", zorder=8,
                    )
                    txt.set_path_effects([
                        patheffects.withStroke(linewidth=2.0, foreground="white", alpha=0.9),
                    ])

        # Title
        prefix = settings.get("title_prefix", "").strip()
        title_core = f"Concentration of {metal_symbol}"
        if metal_unit:
            title_core += f" ({metal_unit})"
        if prefix:
            title_core = f"{prefix} {title_core}"

        ax.text(
            0.46, 0.945, title_core,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=11.5 if is_subpanel else 13.5,
            fontweight="bold", family="DejaVu Sans", zorder=12,
        )

        # 8-point GIS compass rose
        _draw_gis_compass_rose(ax, 0.88, 0.865, size=0.075, aspect_ratio=box_aspect)

        # Panel letter
        active_panel = panel_label if panel_label is not None else settings.get("panel_label", "(a)")
        if active_panel:
            ax.text(
                0.85, 0.67, active_panel,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11.5 if is_subpanel else 13.0,
                fontweight="bold", family="DejaVu Sans", zorder=12,
            )

        # Boxed GIS Legend on the right
        leg_x = 0.745
        leg_y_top = 0.52

        leg_bg = mpatches.Rectangle(
            (leg_x - 0.015, leg_y_top - 0.44), 0.23, 0.47,
            facecolor="white", edgecolor="none", alpha=0.96,
            transform=ax.transAxes, zorder=11,
        )
        ax.add_patch(leg_bg)

        ax.text(
            leg_x, leg_y_top, "Legend", transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=8.5 if is_subpanel else 9.5,
            fontweight="bold", family="DejaVu Sans", zorder=12,
        )
        ax.scatter(
            [leg_x + 0.022], [leg_y_top - 0.038], transform=ax.transAxes,
            s=22 if is_subpanel else 26, c="#7a0118", edgecolors="#2b0008", linewidths=0.5, zorder=12,
        )
        ax.text(
            leg_x + 0.052, leg_y_top - 0.038, "sp_soil", transform=ax.transAxes,
            ha="left", va="center",
            fontsize=7.2 if is_subpanel else 8.0,
            family="DejaVu Sans", zorder=12,
        )
        ax.text(
            leg_x, leg_y_top - 0.078, "<VALUE>", transform=ax.transAxes,
            ha="left", va="center",
            fontsize=7.4 if is_subpanel else 8.2,
            fontweight="bold", family="DejaVu Sans", zorder=12,
        )

        box_w = 0.058
        box_h = 0.027
        row_step = 0.036
        for i in range(9):
            by = leg_y_top - 0.118 - i * row_step
            color_val = cmap(norm_b((bounds[i] + bounds[i + 1]) / 2.0))
            rect = mpatches.Rectangle(
                (leg_x, by), box_w, box_h,
                facecolor=color_val, edgecolor="#555555", linewidth=0.5,
                transform=ax.transAxes, zorder=12,
            )
            ax.add_patch(rect)
            lbl_range = f"{bounds[i]:.2f} - {bounds[i+1]:.2f}"
            ax.text(
                leg_x + box_w + 0.014, by + box_h / 2.0, lbl_range,
                transform=ax.transAxes, ha="left", va="center",
                fontsize=6.5 if is_subpanel else 7.2,
                family="DejaVu Sans", zorder=12,
            )

        # GIS bracket scale bar
        _draw_gis_bracket_scale_bar(ax, gis_x_min, gis_x_max, gis_y_min, gis_y_max)

        # Graticule with DMS formatting on all 4 sides
        wgs84 = pyproj.CRS.from_epsg(4326)
        inv_transformer = pyproj.Transformer.from_crs(target_crs, wgs84, always_xy=True)
        fwd_transformer = pyproj.Transformer.from_crs(wgs84, target_crs, always_xy=True)

        corners_x = [gis_x_min, gis_x_max, gis_x_min, gis_x_max]
        corners_y = [gis_y_min, gis_y_min, gis_y_max, gis_y_max]
        c_lons, c_lats = inv_transformer.transform(corners_x, corners_y)
        g_lon_min, g_lon_max = float(np.min(c_lons)), float(np.max(c_lons))
        g_lat_min, g_lat_max = float(np.min(c_lats)), float(np.max(c_lats))

        def _nice_ticks(vmin: float, vmax: float, max_ticks: int = 5) -> np.ndarray:
            span = vmax - vmin
            if span <= 0:
                return np.array([(vmin + vmax) / 2])
            raw = span / max_ticks
            mag = 10 ** math.floor(math.log10(raw))
            res = raw / mag
            nice = 1 if res <= 1 else (2 if res <= 2 else (5 if res <= 5 else 10))
            step = nice * mag
            start = math.ceil(vmin / step) * step
            ticks = []
            v = start
            while v <= vmax:
                ticks.append(v)
                v += step
            return np.array(ticks) if ticks else np.array([(vmin + vmax) / 2])

        lon_ticks = _nice_ticks(g_lon_min, g_lon_max, 4)
        lat_ticks = _nice_ticks(g_lat_min, g_lat_max, 3)

        mid_lat = (g_lat_min + g_lat_max) / 2
        xt_proj, _ = fwd_transformer.transform(lon_ticks, np.full_like(lon_ticks, mid_lat))
        mid_lon = (g_lon_min + g_lon_max) / 2
        _, yt_proj = fwd_transformer.transform(np.full_like(lat_ticks, mid_lon), lat_ticks)

        xt_mask = (xt_proj >= gis_x_min) & (xt_proj <= gis_x_max)
        yt_mask = (yt_proj >= gis_y_min) & (yt_proj <= gis_y_max)

        ax.set_xticks(xt_proj[xt_mask])
        ax.set_xticklabels([format_dms_lon(x) for x in lon_ticks[xt_mask]], fontsize=6.8 if is_subpanel else 7.5, fontweight="bold")
        ax.set_yticks(yt_proj[yt_mask])
        ax.set_yticklabels([format_dms_lat(y) for y in lat_ticks[yt_mask]], fontsize=6.8 if is_subpanel else 7.5, fontweight="bold", rotation=90, va="center")

        ax.tick_params(axis="x", top=True, bottom=True, labeltop=True, labelbottom=True, direction="out", length=3.5, width=0.85, pad=3.0)
        ax.tick_params(axis="y", left=True, right=True, labelleft=True, labelright=True, direction="out", length=3.5, width=0.85, pad=3.0)
        for label in ax.yaxis.get_ticklabels(which="both"):
            label.set_rotation(90)
            label.set_verticalalignment("center")

        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("black")

    else:
        # Standard Continuous Surface layout
        ax.set_facecolor("white")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")

        im = ax.pcolormesh(
            GX, GY, Z_masked,
            cmap=cmap, norm=norm, alpha=opacity,
            shading="auto", zorder=2,
        )

        show_points = settings.get("show_points", True)
        point_size = settings.get("point_size", 40)
        color_by_value = settings.get("color_by_value", False)

        if show_points:
            if color_by_value:
                ax.scatter(
                    sample_x, sample_y,
                    s=point_size, c=sample_z, cmap=cmap, norm=norm,
                    edgecolors="black", linewidths=0.5, zorder=6,
                    label="Sampling site (measured)",
                )
            else:
                ax.scatter(
                    sample_x, sample_y,
                    s=point_size, facecolors="white", edgecolors="black",
                    linewidths=0.5, zorder=6,
                    label="Sampling site (measured)",
                )
            ax.legend(
                loc="lower right", fontsize=7, framealpha=0.9,
                markerscale=0.8, handletextpad=0.3,
            )

        show_labels = settings.get("show_labels", False)
        if show_labels and sample_labels is not None:
            n_labels = min(len(sample_labels), len(sample_x))
            if n_labels <= MAX_LABELS:
                for i in range(n_labels):
                    lbl = str(sample_labels[i])[:18].replace("$", r"\$")
                    txt = ax.annotate(
                        lbl,
                        (sample_x[i], sample_y[i]),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=7, family="DejaVu Sans", zorder=8,
                    )
                    txt.set_path_effects([
                        patheffects.withStroke(linewidth=2, foreground="white"),
                    ])

        wgs84 = pyproj.CRS.from_epsg(4326)
        inv_transformer = pyproj.Transformer.from_crs(target_crs, wgs84, always_xy=True)
        fwd_transformer = pyproj.Transformer.from_crs(wgs84, target_crs, always_xy=True)

        corners_x = [x_min, x_max, x_min, x_max]
        corners_y = [y_min, y_min, y_max, y_max]
        c_lons, c_lats = inv_transformer.transform(corners_x, corners_y)
        g_lon_min, g_lon_max = float(np.min(c_lons)), float(np.max(c_lons))
        g_lat_min, g_lat_max = float(np.min(c_lats)), float(np.max(c_lats))

        def _nice_ticks_c(vmin: float, vmax: float, max_ticks: int = 6) -> np.ndarray:
            span = vmax - vmin
            if span <= 0:
                return np.array([(vmin + vmax) / 2])
            raw = span / max_ticks
            mag = 10 ** math.floor(math.log10(raw))
            residual = raw / mag
            nice = 1 if residual <= 1 else (2 if residual <= 2 else (5 if residual <= 5 else 10))
            step = nice * mag
            start = math.ceil(vmin / step) * step
            ticks = []
            v = start
            while v <= vmax:
                ticks.append(v)
                v += step
            return np.array(ticks) if ticks else np.array([(vmin + vmax) / 2])

        lon_ticks = _nice_ticks_c(g_lon_min, g_lon_max, 5)
        lat_ticks = _nice_ticks_c(g_lat_min, g_lat_max, 5)

        mid_lat = (g_lat_min + g_lat_max) / 2
        xt_proj, _ = fwd_transformer.transform(lon_ticks, np.full_like(lon_ticks, mid_lat))
        mid_lon = (g_lon_min + g_lon_max) / 2
        _, yt_proj = fwd_transformer.transform(np.full_like(lat_ticks, mid_lon), lat_ticks)

        xt_mask = (xt_proj >= x_min) & (xt_proj <= x_max)
        yt_mask = (yt_proj >= y_min) & (yt_proj <= y_max)

        ax.set_xticks(xt_proj[xt_mask])
        ax.set_xticklabels([f"{v:.4f}°" if abs(v) < 1 else f"{v:.3f}°" for v in lon_ticks[xt_mask]], fontsize=7, family="DejaVu Sans")
        ax.set_yticks(yt_proj[yt_mask])
        ax.set_yticklabels([f"{v:.4f}°" if abs(v) < 1 else f"{v:.3f}°" for v in lat_ticks[yt_mask]], fontsize=7, family="DejaVu Sans")
        ax.tick_params(direction="out", length=4, width=0.5, labelsize=7)

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

        prefix = settings.get("title_prefix", "").strip()
        element_full = ELEMENT_NAMES.get(metal_symbol, "")
        title = f"{metal_symbol} ({element_full}) Concentration" if element_full else f"{metal_symbol} Concentration"
        if prefix:
            title = f"{prefix} {title}"

        if is_subpanel:
            ax.set_title(title, fontsize=11, fontweight="bold", family="DejaVu Sans", pad=8)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, extend=extend)
            unit_label = metal_unit if metal_unit else "Concentration"
            cbar.set_label(unit_label, fontsize=7.5, family="DejaVu Sans")
            cbar.ax.tick_params(labelsize=6.5)
        else:
            fig.text(0.5, 0.95, title, ha="center", va="top", fontsize=16, fontweight="bold", family="DejaVu Sans")
            orientation = settings.get("orientation", "landscape")
            cbar_rect = [0.75, 0.12, 0.02, 0.72] if orientation == "landscape" else [0.87, 0.15, 0.02, 0.60]
            cbar_ax = fig.add_axes(cbar_rect)
            cbar = fig.colorbar(im, cax=cbar_ax, extend=extend)
            unit_label = metal_unit if metal_unit else "Concentration"
            cbar.set_label(unit_label, fontsize=9, family="DejaVu Sans")
            cbar.ax.tick_params(labelsize=7)
            if constant_value:
                cbar.ax.set_title(f"All values = {float(np.mean(sample_z)):.4g}", fontsize=7, pad=4)

        orientation = settings.get("orientation", "landscape")
        if orientation == "landscape":
            _draw_north_arrow(ax, 0.95, 0.90, size=0.06)
        else:
            _draw_north_arrow(ax, 0.93, 0.92, size=0.05)

        x_range_m = x_max - x_min
        bar_length_m, bar_label = _compute_scale_bar(x_range_m)
        sb_x0 = x_min + 0.05 * (x_max - x_min)
        sb_y0 = y_min + 0.05 * (y_max - y_min)
        bar_h = 0.008 * (y_max - y_min)
        ax.add_patch(mpatches.Rectangle((sb_x0, sb_y0), bar_length_m, bar_h, facecolor="black", edgecolor="black", linewidth=0.5, zorder=20))
        ax.add_patch(mpatches.Rectangle((sb_x0 + bar_length_m / 2, sb_y0), bar_length_m / 2, bar_h, facecolor="white", edgecolor="black", linewidth=0.5, zorder=21))
        ax.text(sb_x0 + bar_length_m / 2, sb_y0 - bar_h * 1.5, bar_label, ha="center", va="top", fontsize=7, family="DejaVu Sans", zorder=20)
        ax.text(sb_x0, sb_y0 - bar_h * 1.5, "0", ha="center", va="top", fontsize=7, family="DejaVu Sans", zorder=20)


def render_heatmap(
    metal_symbol: str,
    metal_unit: str,
    Z_masked: np.ma.MaskedArray,
    GX: np.ndarray,
    GY: np.ndarray,
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    sample_z: np.ndarray,
    sample_labels: np.ndarray | None,
    proj_extent: tuple[float, float, float, float],
    target_crs: pyproj.CRS,
    crs_label: str,
    settings: dict[str, Any],
    rmse: float,
    mae: float,
    n_points: int,
    dpi: int = 300,
) -> mfigure.Figure:
    """Render a single A4 heatmap figure."""
    orientation = settings.get("orientation", "landscape")
    if orientation == "landscape":
        fig_w, fig_h = 3508 / 300, 2480 / 300
    else:
        fig_w, fig_h = 2480 / 300, 3508 / 300

    fig = mfigure.Figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="white")
    canvas = FigureCanvasAgg(fig)

    map_style = settings.get("map_style", "GIS Journal / Reference Style (Publication)")
    cmap_name = settings.get("cmap", "gis_yellow_brown")
    is_gis_style = (
        map_style == "GIS Journal / Reference Style (Publication)"
        or settings.get("style") == "gis_publication"
        or cmap_name == "gis_yellow_brown"
    )

    if is_gis_style:
        if orientation == "landscape":
            map_rect = [0.08, 0.08, 0.84, 0.84]
        else:
            map_rect = [0.09, 0.09, 0.82, 0.82]
    else:
        if orientation == "landscape":
            map_rect = [0.08, 0.12, 0.65, 0.72]
        else:
            map_rect = [0.10, 0.15, 0.75, 0.60]

    ax = fig.add_axes(map_rect)
    plot_metal_map_axes(
        ax=ax,
        fig=fig,
        metal_symbol=metal_symbol,
        metal_unit=metal_unit,
        Z_masked=Z_masked,
        GX=GX,
        GY=GY,
        sample_x=sample_x,
        sample_y=sample_y,
        sample_z=sample_z,
        sample_labels=sample_labels,
        proj_extent=proj_extent,
        target_crs=target_crs,
        crs_label=crs_label,
        settings=settings,
        rmse=rmse,
        mae=mae,
        n_points=n_points,
        panel_label=settings.get("panel_label", "(a)"),
        is_subpanel=False,
    )

    # Common footnote / metadata text at bottom
    full_extent_mode = settings.get("full_extent", False)
    note_parts = [
        f"Interpolated surface (IDW, power {settings.get('power', 2.0)}, "
        f"resolution {settings.get('resolution', 300)}, {crs_label}, "
        f"{n_points} points, leave-one-out RMSE {rmse:.4g}).",
        "Not measured values.",
    ]
    if full_extent_mode:
        note_parts.append("Values far from samples are weakly constrained.")
    source = settings.get("source_note", "").strip()
    if source:
        note_parts.append(source)

    note_text = " ".join(note_parts)
    if orientation == "landscape":
        fig.text(0.08, 0.03, note_text, fontsize=7.5, family="DejaVu Sans", va="top")
    else:
        fig.text(0.10, 0.04, note_text, fontsize=7.5, family="DejaVu Sans", va="top")

    return fig


def render_combined_panel_map(
    metal_items: list[dict[str, Any]],
    proj_extent: tuple[float, float, float, float],
    target_crs: pyproj.CRS,
    crs_label: str,
    settings: dict[str, Any],
    dpi: int = 300,
) -> mfigure.Figure:
    """Render an N-panel composite figure showing all analysed metals together.

    If an odd number of metals is analysed (e.g. 3, 5, 7), the last panel is
    aligned in the center of the bottom row.
    """
    n_metals = len(metal_items)
    if n_metals == 0:
        fig = mfigure.Figure(figsize=(8, 6), dpi=dpi, facecolor="white")
        FigureCanvasAgg(fig)
        return fig

    if n_metals == 1:
        n_cols = 1
        n_rows = 1
        fig_w, fig_h = 12.0, 8.0
    elif n_metals == 2:
        n_cols = 2
        n_rows = 1
        fig_w, fig_h = 18.0, 8.5
    else:
        n_cols = 2
        n_rows = (n_metals + 1) // 2
        fig_w = 18.0
        fig_h = max(9.0, n_rows * 6.2 + 0.8)

    fig = mfigure.Figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="white")
    canvas = FigureCanvasAgg(fig)

    left_m = 0.04
    right_m = 0.96
    bot_m = 0.05
    top_m = 0.96
    gap_x = 0.05
    gap_y = 0.06

    total_w = right_m - left_m
    total_h = top_m - bot_m

    w_p = (total_w - (n_cols - 1) * gap_x) / n_cols
    h_p = (total_h - (n_rows - 1) * gap_y) / n_rows

    for idx, item in enumerate(metal_items):
        r = idx // n_cols
        c = idx % n_cols
        row_from_bot = n_rows - 1 - r
        y = bot_m + row_from_bot * (h_p + gap_y)

        # Align odd last panel in center
        is_odd_last = (idx == n_metals - 1) and (n_metals % 2 == 1) and (n_cols == 2) and (n_metals > 1)
        if is_odd_last:
            x = (left_m + right_m - w_p) / 2.0
        else:
            x = left_m + c * (w_p + gap_x)

        ax = fig.add_axes([x, y, w_p, h_p])
        letter = f"({chr(ord('a') + idx)})"

        plot_metal_map_axes(
            ax=ax,
            fig=fig,
            metal_symbol=item["symbol"],
            metal_unit=item.get("unit", ""),
            Z_masked=item["Z_masked"],
            GX=item["GX"],
            GY=item["GY"],
            sample_x=item["sample_x"],
            sample_y=item["sample_y"],
            sample_z=item["sample_z"],
            sample_labels=item.get("sample_labels"),
            proj_extent=proj_extent,
            target_crs=target_crs,
            crs_label=crs_label,
            settings=settings,
            rmse=item.get("rmse", 0.0),
            mae=item.get("mae", 0.0),
            n_points=item.get("n_points", len(item["sample_z"])),
            panel_label=letter,
            is_subpanel=True,
        )

    symbols_str = ", ".join([m["symbol"] for m in metal_items])
    caption = (
        f"Multi-panel spatial distribution of heavy metals ({symbols_str}). "
        f"Interpolated via IDW (power {settings.get('power', 2.0)}, {crs_label}). Not measured values."
    )
    fig.text(0.5, 0.015, caption, ha="center", va="bottom", fontsize=9.0, family="DejaVu Sans", color="#222222")

    return fig


def export_png(fig: mfigure.Figure, dpi: int = 300) -> bytes:
    """Export figure to PNG bytes at the given DPI."""
    buf = io.BytesIO()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    fig.savefig(
        buf, format="png", dpi=dpi,
        facecolor="white", edgecolor="none",
    )
    buf.seek(0)
    return buf.getvalue()


def export_pdf(fig: mfigure.Figure) -> bytes:
    """Export figure to PDF bytes with TrueType fonts embedded."""
    buf = io.BytesIO()
    # Set font type for embedding
    matplotlib.rcParams["pdf.fonttype"] = 42
    canvas = FigureCanvasPdf(fig)
    canvas.draw()
    fig.savefig(
        buf, format="pdf",
        facecolor="white", edgecolor="none",
    )
    buf.seek(0)
    return buf.getvalue()


def create_zip(files: dict[str, bytes]) -> bytes:
    """Create a ZIP file in memory from {filename: bytes}."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    buf.seek(0)
    return buf.getvalue()


def safe_filename(metal_symbol: str) -> str:
    """Reduce a metal symbol to a safe filename component."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", metal_symbol.replace(" ", "_"))
    return safe if safe else "metal"


def sanitize_report_cell(val: Any) -> str:
    """Prefix cells starting with =, +, -, @ with an apostrophe (CSV injection)."""
    s = str(val)
    if s and s[0] in ("=", "+", "-", "@"):
        return "'" + s
    return s


def build_report_csv(report: dict[str, Any]) -> str:
    """Build a downloadable CSV report from the validation results."""
    lines = []
    lines.append("Category,Detail,Value")
    lines.append(f"Rows Read,,{sanitize_report_cell(report.get('rows_read', ''))}")
    lines.append(f"Rows Retained,,{sanitize_report_cell(report.get('rows_retained', ''))}")
    lines.append(f"Rows Excluded,,{sanitize_report_cell(report.get('rows_excluded', ''))}")

    for reason, count in report.get("exclusion_reasons", {}).items():
        lines.append(f"Exclusion Reason,{sanitize_report_cell(reason)},{sanitize_report_cell(count)}")

    for metal_name, metal_counts in report.get("metal_counts", {}).items():
        for key, val in metal_counts.items():
            lines.append(f"Metal: {sanitize_report_cell(metal_name)},{sanitize_report_cell(key)},{sanitize_report_cell(val)}")

    if "extent" in report:
        ext = report["extent"]
        lines.append(f"Extent,Lon range,{sanitize_report_cell(f'{ext[0]:.6f} to {ext[1]:.6f}')}")
        lines.append(f"Extent,Lat range,{sanitize_report_cell(f'{ext[2]:.6f} to {ext[3]:.6f}')}")

    return "\n".join(lines)


def get_metal_title(symbol: str) -> str:
    """Get a display title like 'Pb — Lead Concentration Heatmap'."""
    full_name = ELEMENT_NAMES.get(symbol, "")
    if full_name:
        return f"{symbol} — {full_name} Concentration Heatmap"
    return f"{symbol} Concentration Heatmap"


# =============================================================================
# Streamlit UI (§5)
# =============================================================================


def main() -> None:
    """Streamlit application entry point."""
    import streamlit as st

    st.set_page_config(
        page_title="Heavy Metal Heatmap Generator",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # --- Session state initialisation ---
    if "stage" not in st.session_state:
        st.session_state.stage = "upload"
    if "results" not in st.session_state:
        st.session_state.results = {}
    if "upload_key" not in st.session_state:
        st.session_state.upload_key = 0

    # === Stage A: Upload ===
    st.title("Heavy Metal Heatmap Generator")
    st.markdown(
        "Generate publication-ready concentration heatmaps from CSV data of "
        "geographic sampling points and heavy-metal concentrations. Upload your "
        "data, configure the map settings, and download A4 maps as PNG and PDF."
    )

    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"],
        key=f"csv_uploader_{st.session_state.upload_key}",
    )

    with st.expander("CSV format (illustrative)"):
        st.markdown("""
| Sampling Sites | Longitudes | Latitudes | Pb | Cd | Cr | Cu |
|---|---:|---:|---:|---:|---:|---:|
| Site 1 | 90.412 | 23.810 | 12.5 | 0.8 | 25.4 | 18.2 |
| Site 2 | 90.425 | 23.796 | 15.2 | 1.1 | 21.7 | 16.9 |
| Site 3 | 90.438 | 23.785 | 11.8 | 0.6 | 28.1 | 20.5 |
        """)

    st.download_button(
        "Download CSV template",
        data=CSV_TEMPLATE,
        file_name="heatmap_template.csv",
        mime="text/csv",
    )

    st.caption(
        "Your file is processed in memory for this session only and is not stored."
    )

    if uploaded_file is None:
        st.session_state.stage = "upload"
        return

    # Size check
    file_bytes = uploaded_file.getvalue()
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > MAX_UPLOAD_MB:
        st.error(
            f"File is {file_size_mb:.1f} MB, which exceeds the {MAX_UPLOAD_MB} MB limit. "
            "Please reduce the file size."
        )
        return

    # Parse
    try:
        df = read_csv_flexible(file_bytes)
    except ValueError as e:
        st.error(f"Could not read the CSV file: {e}")
        return

    if len(df) == 0:
        st.error("The uploaded file contains no data rows.")
        return

    if len(df) > MAX_ROWS:
        st.error(
            f"The file has {len(df):,} rows, which exceeds the {MAX_ROWS:,} row limit. "
            "Please reduce the dataset."
        )
        return

    # Column detection
    site_col, lon_col, lat_col, all_cols = detect_columns(df)

    if lon_col is None or lat_col is None:
        missing = []
        if lon_col is None:
            missing.append("Longitude")
        if lat_col is None:
            missing.append("Latitude")
        st.error(
            f"Could not find {' or '.join(missing)} column(s). "
            f"Columns found: {', '.join(all_cols)}. "
            "Please use one of the recognised names: "
            f"Longitude ({', '.join(LON_ALIASES)}), "
            f"Latitude ({', '.join(LAT_ALIASES)})."
        )
        return

    # Site numbering if missing
    site_warning = None
    if site_col is None:
        df["_site"] = [f"Site {i+1}" for i in range(len(df))]
        site_col = "_site"
        site_warning = "No site/sample column found — sites numbered Site 1 to Site N."

    # Coordinate validation
    lon_arr, lat_arr, coord_valid, coord_report = validate_coordinates(
        df[lon_col], df[lat_col],
    )

    if coord_report.get("dms_detected"):
        st.error(
            "Degree-minute-second (DMS) text detected in coordinate columns. "
            "Please convert to decimal degrees (WGS84) before uploading."
        )
        return

    # === Stage B: Review ===
    st.divider()
    st.subheader("📋 Data Review")
    st.markdown(f"**File:** {uploaded_file.name}")

    # Swap suggestion
    if coord_report.get("swap_suggested"):
        swap = st.checkbox(
            "⚠️ Swap longitude and latitude? (All latitudes exceed ±90° "
            "but longitudes fit within ±90°)",
            value=False,
        )
        if swap:
            lon_arr, lat_arr = lat_arr, lon_arr
            coord_valid_new = (
                (np.isfinite(lon_arr)) & (np.isfinite(lat_arr))
                & (lon_arr >= -180) & (lon_arr <= 180)
                & (lat_arr >= -90) & (lat_arr <= 90)
            )
            coord_valid = coord_valid_new
            coord_report["valid"] = int(coord_valid.sum())
            coord_report["excluded"] = int((~coord_valid).sum())

    if coord_report["valid"] == 0:
        st.error(
            "No valid coordinates found. All rows have coordinates outside "
            "the valid range (longitude: -180 to 180, latitude: -90 to 90) "
            "or are non-numeric."
        )
        return

    # Metal detection
    metals = detect_metals(df, site_col, lon_col, lat_col)
    if not metals:
        st.error(
            "No numeric columns detected that could represent metal concentrations. "
            "Ensure your CSV has columns with at least 50% numeric values beyond "
            "the site, longitude, and latitude columns."
        )
        return

    # Metal selection
    pre_selected = [m["name"] for m in metals if m["pre_selected"] and m["usable"]]
    all_metal_names = [m["name"] for m in metals]

    selected_metals = st.multiselect(
        f"Detected metals/analytes ({len(metals)} candidates). Edit selection:",
        options=all_metal_names,
        default=pre_selected,
        help="Select the columns to generate heatmaps for.",
    )

    if len(selected_metals) > MAX_METALS:
        st.error(f"At most {MAX_METALS} metals may be selected. You selected {len(selected_metals)}.")
        return

    if not selected_metals:
        st.warning("Select at least one metal to continue.")
        return

    # Per-metal processing and validation
    valid_lons = lon_arr[coord_valid]
    valid_lats = lat_arr[coord_valid]
    valid_sites = df[site_col].values[coord_valid]

    # Build a report
    report: dict[str, Any] = {
        "rows_read": len(df),
        "rows_retained": int(coord_valid.sum()),
        "rows_excluded": int((~coord_valid).sum()),
        "exclusion_reasons": coord_report.get("reasons", {}),
        "metal_counts": {},
    }

    metal_data: dict[str, dict[str, Any]] = {}
    qualifying_metals: list[str] = []
    blocked = False

    # Units from headers (editable)
    units: dict[str, str] = {}
    for m in metals:
        if m["name"] in selected_metals:
            units[m["name"]] = m["unit"]

    with st.expander("Per-metal units (editable)"):
        for m_name in selected_metals:
            meta = next((m for m in metals if m["name"] == m_name), None)
            symbol = meta["symbol"] if meta else m_name
            default_unit = units.get(m_name, "")
            u = st.text_input(
                f"Unit for {symbol}",
                value=default_unit,
                max_chars=20,
                key=f"unit_{m_name}",
            )
            units[m_name] = u

    for m_name in selected_metals:
        series = df[m_name].values[coord_valid]
        values, counts = process_metal_values(pd.Series(series))

        # Average duplicates
        u_lons, u_lats, u_vals, n_dup = average_duplicates(
            valid_lons, valid_lats, values,
        )

        n_valid = len(u_vals)
        counts["duplicates_averaged"] = n_dup

        report["metal_counts"][m_name] = counts

        meta = next((m for m in metals if m["name"] == m_name), None)
        symbol = meta["symbol"] if meta else m_name

        if n_valid < MIN_POINTS:
            st.warning(
                f"**{symbol}**: Only {n_valid} distinct valid locations "
                f"(minimum {MIN_POINTS} required). This metal will be skipped."
            )
            metal_data[m_name] = {"skip": True, "reason": f"Only {n_valid} points (need {MIN_POINTS})"}
        else:
            if n_valid < WARN_POINTS:
                st.warning(
                    f"**{symbol}**: Only {n_valid} points — the surface "
                    "is weakly constrained."
                )
            metal_data[m_name] = {
                "skip": False,
                "symbol": symbol,
                "unit": units.get(m_name, ""),
                "lons": u_lons,
                "lats": u_lats,
                "values": u_vals,
                "n_valid": n_valid,
                "counts": counts,
                "all_lons": valid_lons[np.isfinite(values)],
                "all_lats": valid_lats[np.isfinite(values)],
                "labels": valid_sites[np.isfinite(values)] if valid_sites is not None else None,
            }
            qualifying_metals.append(m_name)

    if not qualifying_metals:
        st.error(
            "No metal has enough valid data points for mapping "
            f"(minimum {MIN_POINTS} distinct locations required)."
        )
        return

    # Summary stats
    st.markdown(f"**Sampling locations:** {coord_report['valid']}")
    extent_lons = np.concatenate([metal_data[m]["lons"] for m in qualifying_metals])
    extent_lats = np.concatenate([metal_data[m]["lats"] for m in qualifying_metals])
    raw_lon_min, raw_lon_max = float(np.min(extent_lons)), float(np.max(extent_lons))
    raw_lat_min, raw_lat_max = float(np.min(extent_lats)), float(np.max(extent_lats))
    st.markdown(
        f"**Coordinate extent:** {raw_lon_min:.6f}° – {raw_lon_max:.6f}° lon, "
        f"{raw_lat_min:.6f}° – {raw_lat_max:.6f}° lat (WGS84 decimal degrees)"
    )
    st.markdown(f"**Metals for mapping:** {', '.join(qualifying_metals)}")

    if site_warning:
        st.warning(site_warning)

    if coord_report.get("has_origin"):
        st.warning("⚠️ One or more points are at (0, 0) — null island. Please verify these are real locations.")

    exclude_pct = coord_report["excluded"] / max(coord_report["total"], 1) * 100
    if exclude_pct > 20:
        st.warning(
            f"⚠️ {exclude_pct:.0f}% of rows were excluded. "
            "Check your data for coordinate or formatting issues."
        )

    # Report table and download
    with st.expander("Data quality report"):
        report["extent"] = (raw_lon_min, raw_lon_max, raw_lat_min, raw_lat_max)

        # Per-metal counts
        for m_name in selected_metals:
            if m_name in report["metal_counts"]:
                c = report["metal_counts"][m_name]
                st.markdown(
                    f"**{m_name}**: {c.get('valid', 0) + c.get('zero', 0)} valid, "
                    f"{c.get('missing', 0)} missing, "
                    f"{c.get('non_detect', 0)} non-detect, "
                    f"{c.get('negative', 0)} negative, "
                    f"{c.get('invalid', 0)} invalid, "
                    f"{c.get('duplicates_averaged', 0)} duplicate groups averaged"
                )

        # First 20 cleaned rows
        preview_data = []
        for i in range(min(20, len(valid_lons))):
            row = {"Lon": valid_lons[i], "Lat": valid_lats[i]}
            if valid_sites is not None:
                row["Site"] = valid_sites[i]
            for m_name in qualifying_metals:
                md = metal_data[m_name]
                # Original (pre-dedup) values
                vals = df[m_name].values[coord_valid]
                v, _ = _classify_value(vals[i])
                if v in ("valid", "zero"):
                    row[m_name] = float(str(vals[i]))
                else:
                    row[m_name] = None
            preview_data.append(row)
        st.dataframe(pd.DataFrame(preview_data), use_container_width=True)

        report_csv = build_report_csv(report)
        st.download_button(
            "Download report as CSV",
            data=report_csv,
            file_name="data_quality_report.csv",
            mime="text/csv",
        )

    # === Stage C: Settings and Generate ===
    st.divider()
    st.subheader("⚙️ Map Settings")

    with st.form("settings_form"):
        map_style = st.selectbox(
            "Map Presentation Style",
            options=["GIS Journal / Reference Style (Publication)", "Continuous Surface (Modern A4)"],
            index=0,
            help="GIS Journal style renders discrete 9-class contours, 8-point compass rose, bracket scale bar, boxed legend and DMS graticule.",
        )
        col1, col2 = st.columns(2)

        with col1:
            cmap_options = (
                ["gis_yellow_brown (GIS Reference 9-Class)", "viridis", "plasma", "inferno", "magma", "cividis"]
                + ["coolwarm (diverging)", "RdYlBu (diverging)", "Spectral (diverging)"]
            )
            cmap_sel = st.selectbox(
                "Colour ramp",
                options=cmap_options,
                index=0,
                help="Sequential ramps for concentration data; diverging for deviation from a mean.",
            )
            reverse_cmap = st.checkbox("Reverse ramp", value=False, help="Invert the colour direction.")
            colour_scale = st.selectbox(
                "Colour scale",
                options=["linear", "log", "percentile clip"],
                index=0,
                help="Log scale requires all values > 0. Percentile clip uses the 2nd–98th percentile range.",
            )
            resolution = st.slider(
                "Grid resolution",
                min_value=100, max_value=800, value=300, step=10,
                help="Number of cells along the longer side of the interpolation grid.",
            )
            power = st.slider(
                "IDW power",
                min_value=1.0, max_value=6.0, value=2.0, step=0.5,
                help="Higher values give more weight to nearby points (sharper transitions).",
            )
            full_extent = st.checkbox(
                "Full buffered extent (show entire grid)",
                value=False,
                help="By default, cells outside the convex hull of samples are hidden.",
            )

        with col2:
            opacity = st.slider(
                "Opacity", min_value=0.3, max_value=1.0, value=1.0, step=0.05,
                help="Transparency of the interpolated surface.",
            )
            show_points = st.checkbox("Show sampling points", value=True)
            point_size = st.slider(
                "Point size", min_value=10, max_value=200, value=30, step=5,
                help="Marker size for sampling points.",
            )
            color_by_value = st.checkbox(
                "Colour points by measured value", value=False,
                help="Colour markers by concentration instead of white fill.",
            )
            show_labels = st.checkbox(
                "Show site labels", value=True,
                help=f"Label each sampling site (max {MAX_LABELS} sites).",
            )
            panel_label = st.text_input(
                "Panel letter (optional)", value="(a)", max_chars=10,
                help="e.g. (a), (b) for journal multi-panel figures.",
            )
            orientation = st.selectbox(
                "Orientation", options=["landscape", "portrait"], index=0,
                help="A4 page orientation.",
            )
            title_prefix = st.text_input(
                "Title prefix", value="", max_chars=80,
                help="Optional text prepended to each map title.",
            )
            source_note = st.text_input(
                "Source note", value="", max_chars=120,
                help="Optional attribution or source note in the map footer.",
            )

        submitted = st.form_submit_button("Generate Heatmaps")

    if not submitted:
        return

    # Parse cmap selection
    cmap_name = cmap_sel.split(" (")[0].strip()

    settings = {
        "map_style": map_style,
        "panel_label": panel_label,
        "cmap": cmap_name,
        "reverse_cmap": reverse_cmap,
        "colour_scale": colour_scale,
        "resolution": resolution,
        "power": power,
        "full_extent": full_extent,
        "opacity": opacity,
        "show_points": show_points,
        "point_size": point_size,
        "color_by_value": color_by_value,
        "show_labels": show_labels,
        "orientation": orientation,
        "title_prefix": title_prefix,
        "source_note": source_note,
    }

    # === Compute shared extent ===
    all_valid_lons = np.concatenate([metal_data[m]["lons"] for m in qualifying_metals])
    all_valid_lats = np.concatenate([metal_data[m]["lats"] for m in qualifying_metals])
    # Unique for extent
    extent_coords = set(zip(np.round(all_valid_lons, 6), np.round(all_valid_lats, 6)))
    ext_lons = np.array([c[0] for c in extent_coords])
    ext_lats = np.array([c[1] for c in extent_coords])

    lon_min, lon_max, lat_min, lat_max = compute_extent(ext_lons, ext_lats)

    # CRS
    centroid_lon = float(np.mean(ext_lons))
    centroid_lat = float(np.mean(ext_lats))
    target_crs, crs_label, crs_approx = determine_crs(
        centroid_lon, centroid_lat, lon_max - lon_min,
    )
    if crs_approx:
        st.warning(
            f"Using an azimuthal equidistant projection centred on "
            f"({centroid_lon:.3f}°, {centroid_lat:.3f}°). "
            "Distances are approximate."
        )

    # Project extent
    proj_extent = project_extent(lon_min, lon_max, lat_min, lat_max, target_crs)

    # Build grid once
    GX, GY, nx, ny = build_grid(
        proj_extent[0], proj_extent[1],
        proj_extent[2], proj_extent[3],
        resolution,
    )

    # === Process each metal ===
    progress = st.progress(0.0)
    status_text = st.empty()
    results: dict[str, dict[str, Any]] = {}
    metal_items: list[dict[str, Any]] = []
    total = len(qualifying_metals)

    for idx, m_name in enumerate(qualifying_metals):
        md = metal_data[m_name]
        symbol = md["symbol"]
        status_text.text(f"Processing {symbol} ({idx + 1}/{total})…")

        try:
            # Project sample points
            sx, sy = project_points(md["lons"], md["lats"], target_crs)

            # IDW
            Z = idw_interpolate(sx, sy, md["values"], GX, GY, power=power)

            # Mask
            Z_masked = apply_mask(
                Z, GX, GY, sx, sy,
                full_extent=full_extent,
                distance_multiplier=3.0,
            )

            # LOO CV
            rmse, mae = leave_one_out_cv(sx, sy, md["values"], power=power)

            # Get labels for this metal
            labels_arr = md.get("labels", None)

            # Store item for combined multi-panel map
            metal_items.append({
                "symbol": symbol,
                "unit": md["unit"],
                "Z_masked": Z_masked,
                "GX": GX,
                "GY": GY,
                "sample_x": sx,
                "sample_y": sy,
                "sample_z": md["values"],
                "sample_labels": labels_arr,
                "rmse": rmse,
                "mae": mae,
                "n_points": md["n_valid"],
            })

            # Render at 100 DPI for preview
            fig_preview = render_heatmap(
                metal_symbol=symbol,
                metal_unit=md["unit"],
                Z_masked=Z_masked,
                GX=GX, GY=GY,
                sample_x=sx, sample_y=sy,
                sample_z=md["values"],
                sample_labels=labels_arr,
                proj_extent=proj_extent,
                target_crs=target_crs,
                crs_label=crs_label,
                settings=settings,
                rmse=rmse, mae=mae,
                n_points=md["n_valid"],
                dpi=100,
            )
            preview_bytes = export_png(fig_preview, dpi=100)
            fig_preview.clear()
            del fig_preview

            # Render at 300 DPI for export
            fig_export = render_heatmap(
                metal_symbol=symbol,
                metal_unit=md["unit"],
                Z_masked=Z_masked,
                GX=GX, GY=GY,
                sample_x=sx, sample_y=sy,
                sample_z=md["values"],
                sample_labels=labels_arr,
                proj_extent=proj_extent,
                target_crs=target_crs,
                crs_label=crs_label,
                settings=settings,
                rmse=rmse, mae=mae,
                n_points=md["n_valid"],
                dpi=300,
            )
            png_bytes = export_png(fig_export, dpi=300)
            pdf_bytes = export_pdf(fig_export)
            fig_export.clear()
            del fig_export

            safe_name = safe_filename(symbol)
            results[m_name] = {
                "symbol": symbol,
                "preview": preview_bytes,
                "png": png_bytes,
                "pdf": pdf_bytes,
                "png_name": f"{safe_name}_Heatmap.png",
                "pdf_name": f"{safe_name}_Heatmap.pdf",
                "n_valid": md["n_valid"],
                "min": float(np.min(md["values"])),
                "mean": float(np.mean(md["values"])),
                "max": float(np.max(md["values"])),
                "rmse": rmse,
                "mae": mae,
                "error": None,
            }

        except Exception as e:
            results[m_name] = {
                "symbol": symbol,
                "error": str(e),
            }

        progress.progress((idx + 1) / total)

    # Render combined multi-panel map if multiple metals are present
    combined_result = None
    if len(metal_items) >= 2:
        status_text.text("Generating combined multi-panel map…")
        fig_combined_prev = render_combined_panel_map(
            metal_items, proj_extent, target_crs, crs_label, settings, dpi=100,
        )
        combined_prev_bytes = export_png(fig_combined_prev, dpi=100)
        fig_combined_prev.clear()
        del fig_combined_prev

        fig_combined_export = render_combined_panel_map(
            metal_items, proj_extent, target_crs, crs_label, settings, dpi=300,
        )
        combined_png_bytes = export_png(fig_combined_export, dpi=300)
        combined_pdf_bytes = export_pdf(fig_combined_export)
        fig_combined_export.clear()
        del fig_combined_export

        combined_result = {
            "preview": combined_prev_bytes,
            "png": combined_png_bytes,
            "pdf": combined_pdf_bytes,
            "n_panels": len(metal_items),
        }

    # Resolve filename collisions
    used_names: set[str] = set()
    for m_name in results:
        r = results[m_name]
        if r.get("error"):
            continue
        for key in ("png_name", "pdf_name"):
            name = r[key]
            if name in used_names:
                base, ext = name.rsplit(".", 1)
                counter = 2
                while f"{base}_{counter}.{ext}" in used_names:
                    counter += 1
                name = f"{base}_{counter}.{ext}"
                r[key] = name
            used_names.add(name)

    status_text.text("Done!")
    st.session_state.results = results
    st.session_state.combined_result = combined_result

    # === Stage D: Gallery and Downloads ===
    st.divider()

    # Collect successes for ZIP
    zip_files: dict[str, bytes] = {}
    failures: list[str] = []

    # Display combined multi-panel map first if available
    if combined_result is not None:
        st.subheader(f"🖼️ Combined Multi-Panel Publication Map ({combined_result['n_panels']} Metals)")
        st.markdown(
            "Publication-ready composite figure with all metals "
            "(panels (a), (b), ... with odd last panel aligned center)."
        )
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.download_button(
                "⬇️ Download Combined PNG (300 DPI)",
                data=combined_result["png"],
                file_name="All_Metals_Combined_Panel_Map.png",
                mime="image/png",
                key="dl_comb_png",
            )
        with col_c2:
            st.download_button(
                "⬇️ Download Combined PDF (Vector)",
                data=combined_result["pdf"],
                file_name="All_Metals_Combined_Panel_Map.pdf",
                mime="application/pdf",
                key="dl_comb_pdf",
            )
        st.image(
            combined_result["preview"],
            caption=f"Combined {combined_result['n_panels']}-Panel Publication Map",
            use_container_width=True,
        )
        zip_files["All_Metals_Combined_Panel_Map.png"] = combined_result["png"]
        zip_files["All_Metals_Combined_Panel_Map.pdf"] = combined_result["pdf"]
        st.divider()

    st.subheader("🗺️ Individual Metal Heatmaps")

    first = True
    for m_name in qualifying_metals:
        r = results.get(m_name)
        if r is None:
            continue

        symbol = r.get("symbol", m_name)
        title = get_metal_title(symbol)

        if r.get("error"):
            failures.append(f"{symbol}: {r['error']}")
            st.error(f"**{symbol}** failed: {r['error']}")
            continue

        with st.expander(title, expanded=first):
            st.image(r["preview"], use_container_width=True)
            st.markdown(
                f"**{symbol}**: {r['n_valid']} valid points | "
                f"Min: {r['min']:.4g} | Mean: {r['mean']:.4g} | "
                f"Max: {r['max']:.4g} | LOO RMSE: {r['rmse']:.4g}"
            )

            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    f"Download PNG",
                    data=r["png"],
                    file_name=r["png_name"],
                    mime="image/png",
                    key=f"dl_png_{m_name}",
                )
            with dl_col2:
                st.download_button(
                    f"Download PDF",
                    data=r["pdf"],
                    file_name=r["pdf_name"],
                    mime="application/pdf",
                    key=f"dl_pdf_{m_name}",
                )

            zip_files[r["png_name"]] = r["png"]
            zip_files[r["pdf_name"]] = r["pdf"]

        first = False

    if failures:
        st.warning("**Failed metals:** " + "; ".join(failures))

    if zip_files:
        zip_bytes = create_zip(zip_files)
        st.download_button(
            "📦 Download All Heatmaps",
            data=zip_bytes,
            file_name="Heatmaps.zip",
            mime="application/zip",
            key="dl_zip",
        )

    with st.expander("About these maps"):
        st.markdown(
            "The surface shown is an **Inverse Distance Weighted (IDW) estimate** "
            "computed from nearby sampling points. It is **not** a direct measurement "
            "at every location. The interpolation ignores geology, pollution sources, "
            "and measurement uncertainty. The estimate is weakest where samples are "
            "sparse or absent. These maps are for exploratory analysis only — validate "
            "results against field data before regulatory or health decisions."
        )

    # Start new analysis button
    if st.button("🔄 Start new analysis"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.upload_key = st.session_state.get("upload_key", 0) + 1
        st.session_state.stage = "upload"
        st.session_state.results = {}
        st.rerun()


if __name__ == "__main__":
    main()
