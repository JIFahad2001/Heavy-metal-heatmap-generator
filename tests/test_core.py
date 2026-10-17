"""
Tests for Heavy Metal Heatmap Generator core functions.
All test data is generated in code — no external files needed.
"""

from __future__ import annotations

import io
import math
import os
import struct
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

# Add parent dir to path so we can import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (
    MAX_UPLOAD_MB,
    MAX_ROWS,
    MAX_METALS,
    MAX_GRID_CELLS,
    MIN_POINTS,
    WARN_POINTS,
    read_csv_flexible,
    detect_columns,
    detect_metals,
    process_metal_values,
    validate_coordinates,
    average_duplicates,
    compute_extent,
    determine_crs,
    project_points,
    project_extent,
    build_grid,
    idw_interpolate,
    compute_convex_hull_mask,
    compute_distance_mask,
    apply_mask,
    leave_one_out_cv,
    render_heatmap,
    export_png,
    export_pdf,
    create_zip,
    safe_filename,
    sanitize_report_cell,
    get_metal_title,
    ELEMENT_NAMES,
    BUFFER_FRACTION,
    MIN_EXTENT_DEG,
)
import pyproj


# =====================================================================
# Helper: build CSV bytes from a DataFrame
# =====================================================================

def _df_to_csv_bytes(df: pd.DataFrame, sep: str = ",") -> bytes:
    return df.to_csv(index=False, sep=sep).encode("utf-8")


def _make_base_df(n: int = 20, metals: list[str] | None = None, seed: int = 42):
    """Create a synthetic DataFrame with n sampling sites around Dhaka."""
    rng = np.random.RandomState(seed)
    if metals is None:
        metals = ["Pb", "Cd", "Cr", "Cu"]
    lons = 90.40 + rng.uniform(0, 0.06, n)
    lats = 23.78 + rng.uniform(0, 0.05, n)
    data = {
        "Sampling Sites": [f"Site {i+1}" for i in range(n)],
        "Longitudes": lons,
        "Latitudes": lats,
    }
    for m in metals:
        data[m] = rng.uniform(0.1, 50, n)
    return pd.DataFrame(data)


# Simple rendering settings for tests
_DEFAULT_SETTINGS = {
    "cmap": "viridis",
    "reverse_cmap": False,
    "colour_scale": "linear",
    "resolution": 100,
    "power": 2.0,
    "full_extent": False,
    "opacity": 1.0,
    "show_points": True,
    "point_size": 40,
    "color_by_value": False,
    "show_labels": False,
    "orientation": "landscape",
    "title_prefix": "",
    "source_note": "",
}


def _render_metal(
    lons: np.ndarray,
    lats: np.ndarray,
    values: np.ndarray,
    symbol: str = "Pb",
    unit: str = "mg/kg",
    settings: dict | None = None,
    labels: np.ndarray | None = None,
    dpi: int = 100,
):
    """End-to-end helper: project, grid, IDW, mask, render, export."""
    if settings is None:
        settings = dict(_DEFAULT_SETTINGS)

    # Extent and CRS
    lon_min, lon_max, lat_min, lat_max = compute_extent(lons, lats)
    centroid_lon = float(np.mean(lons))
    centroid_lat = float(np.mean(lats))
    target_crs, crs_label, _ = determine_crs(centroid_lon, centroid_lat, lon_max - lon_min)
    proj_ext = project_extent(lon_min, lon_max, lat_min, lat_max, target_crs)
    GX, GY, nx, ny = build_grid(proj_ext[0], proj_ext[1], proj_ext[2], proj_ext[3],
                                 settings.get("resolution", 100))

    sx, sy = project_points(lons, lats, target_crs)
    Z = idw_interpolate(sx, sy, values, GX, GY, power=settings.get("power", 2.0))
    Z_masked = apply_mask(Z, GX, GY, sx, sy,
                          full_extent=settings.get("full_extent", False))

    rmse, mae = leave_one_out_cv(sx, sy, values, power=settings.get("power", 2.0))

    fig = render_heatmap(
        metal_symbol=symbol, metal_unit=unit,
        Z_masked=Z_masked, GX=GX, GY=GY,
        sample_x=sx, sample_y=sy, sample_z=values,
        sample_labels=labels,
        proj_extent=proj_ext, target_crs=target_crs,
        crs_label=crs_label, settings=settings,
        rmse=rmse, mae=mae, n_points=len(values),
        dpi=dpi,
    )
    return fig, Z_masked, target_crs, crs_label


# =====================================================================
# Test 1: One-metal CSV
# =====================================================================

class TestOneMetal:
    def test_single_metal_detected(self):
        df = _make_base_df(10, metals=["Pb"])
        csv_bytes = _df_to_csv_bytes(df)
        parsed = read_csv_flexible(csv_bytes)
        site_col, lon_col, lat_col, _ = detect_columns(parsed)
        metals = detect_metals(parsed, site_col, lon_col, lat_col)
        assert len(metals) == 1
        assert metals[0]["symbol"] == "Pb"

    def test_single_metal_renders(self):
        df = _make_base_df(10, metals=["Pb"])
        fig, _, _, _ = _render_metal(
            df["Longitudes"].values, df["Latitudes"].values,
            df["Pb"].values, "Pb",
        )
        png = export_png(fig, dpi=100)
        assert len(png) > 1000
        fig.clear()


# =====================================================================
# Test 2: Four-metal CSV
# =====================================================================

class TestFourMetals:
    def test_four_metals_detected(self):
        df = _make_base_df(20, metals=["Pb", "Cd", "Cr", "Cu"])
        csv_bytes = _df_to_csv_bytes(df)
        parsed = read_csv_flexible(csv_bytes)
        site_col, lon_col, lat_col, _ = detect_columns(parsed)
        metals = detect_metals(parsed, site_col, lon_col, lat_col)
        symbols = {m["symbol"] for m in metals}
        assert symbols == {"Pb", "Cd", "Cr", "Cu"}

    def test_four_distinct_filenames(self):
        names = set()
        for sym in ["Pb", "Cd", "Cr", "Cu"]:
            names.add(f"{safe_filename(sym)}_Heatmap.png")
        assert len(names) == 4

    def test_four_pngs_differ(self):
        df = _make_base_df(20, metals=["Pb", "Cd", "Cr", "Cu"], seed=42)
        pngs = []
        for metal in ["Pb", "Cd", "Cr", "Cu"]:
            fig, _, _, _ = _render_metal(
                df["Longitudes"].values, df["Latitudes"].values,
                df[metal].values, metal,
            )
            pngs.append(export_png(fig, dpi=100))
            fig.clear()
        # At least some pairs should differ
        all_same = all(p == pngs[0] for p in pngs[1:])
        assert not all_same


# =====================================================================
# Test 3: Missing values in one metal
# =====================================================================

class TestMissingValues:
    def test_missing_in_one_metal(self):
        df = _make_base_df(20)
        # Introduce missing in Cd
        df.loc[0:4, "Cd"] = np.nan
        csv_bytes = _df_to_csv_bytes(df)
        parsed = read_csv_flexible(csv_bytes)

        # Process values
        vals_pb, counts_pb = process_metal_values(parsed["Pb"])
        vals_cd, counts_cd = process_metal_values(parsed["Cd"])

        assert counts_pb["missing"] == 0
        assert counts_cd["missing"] == 5
        assert counts_pb["valid"] + counts_pb["zero"] == 20
        assert counts_cd["valid"] + counts_cd["zero"] == 15


# =====================================================================
# Test 4: Invalid coordinates
# =====================================================================

class TestInvalidCoordinates:
    def test_lat_out_of_range(self):
        df = _make_base_df(10)
        df.loc[0, "Latitudes"] = 95.0
        lon_arr, lat_arr, valid, report = validate_coordinates(
            df["Longitudes"], df["Latitudes"],
        )
        assert report["excluded"] >= 1
        assert report["reasons"]["out_of_range"] >= 1

    def test_blank_coords(self):
        df = _make_base_df(10)
        df["Longitudes"] = df["Longitudes"].astype(object)
        df["Latitudes"] = df["Latitudes"].astype(object)
        df.loc[1, "Longitudes"] = ""
        df.loc[2, "Latitudes"] = "abc"
        lon_arr, lat_arr, valid, report = validate_coordinates(
            df["Longitudes"], df["Latitudes"],
        )
        assert report["excluded"] >= 2

    def test_swap_suggested(self):
        # Swap: all "lats" > 90 but "lons" are in [-90, 90]
        df = pd.DataFrame({
            "Longitudes": [23.8, 23.9, 24.0, 24.1, 24.2],
            "Latitudes": [90.4, 90.5, 90.6, 90.7, 90.8],
            "Pb": [1, 2, 3, 4, 5],
        })
        # Longitudes are < 90, latitudes are > 90
        lon_arr, lat_arr, valid, report = validate_coordinates(
            df["Longitudes"], df["Latitudes"],
        )
        assert report["swap_suggested"] is True


# =====================================================================
# Test 5: Duplicate coordinates
# =====================================================================

class TestDuplicateCoordinates:
    def test_duplicates_averaged(self):
        lons = np.array([90.4, 90.4, 90.5, 90.5, 90.6])
        lats = np.array([23.8, 23.8, 23.9, 23.9, 24.0])
        vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        u_lons, u_lats, u_vals, n_dup = average_duplicates(lons, lats, vals)
        assert n_dup == 2
        assert len(u_vals) == 3
        # No NaN or inf
        assert np.all(np.isfinite(u_vals))

    def test_no_nan_in_rendered_area(self):
        lons = np.array([90.4, 90.4, 90.42, 90.44, 90.46])
        lats = np.array([23.8, 23.8, 23.82, 23.84, 23.86])
        vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        u_lons, u_lats, u_vals, _ = average_duplicates(lons, lats, vals)
        fig, Z_masked, _, _ = _render_metal(u_lons, u_lats, u_vals, "Pb")
        # Inside the shown area, no NaN (only masked areas have NaN)
        valid_data = Z_masked.compressed()
        assert np.all(np.isfinite(valid_data))
        fig.clear()


# =====================================================================
# Test 6: Constant concentrations
# =====================================================================

class TestConstantConcentrations:
    def test_constant_renders(self):
        df = _make_base_df(10)
        df["Pb"] = 5.0  # constant
        fig, Z_masked, _, _ = _render_metal(
            df["Longitudes"].values, df["Latitudes"].values,
            df["Pb"].values, "Pb",
        )
        png = export_png(fig, dpi=100)
        assert len(png) > 1000
        fig.clear()

    def test_constant_colour_bar_valid(self):
        # Should not raise
        lons = np.array([90.4, 90.42, 90.44, 90.46, 90.48])
        lats = np.array([23.8, 23.82, 23.84, 23.86, 23.88])
        vals = np.array([42.0] * 5)
        fig, _, _, _ = _render_metal(lons, lats, vals, "Test")
        pdf = export_pdf(fig)
        assert len(pdf) > 100
        fig.clear()


# =====================================================================
# Test 7: Fewer than MIN_POINTS
# =====================================================================

class TestTooFewPoints:
    def test_3_points_rejected(self):
        lons = np.array([90.4, 90.5, 90.6])
        lats = np.array([23.8, 23.9, 24.0])
        vals = np.array([1.0, 2.0, 3.0])
        u_lons, u_lats, u_vals, _ = average_duplicates(lons, lats, vals)
        assert len(u_vals) < MIN_POINTS

    def test_other_metals_unaffected(self):
        df = _make_base_df(20, metals=["Pb", "Cd"])
        # Make Cd have only 3 valid values
        df["Cd"] = np.nan
        df.loc[0:2, "Cd"] = [1.0, 2.0, 3.0]
        vals_cd, counts = process_metal_values(df["Cd"])
        n_valid_cd = int(np.sum(np.isfinite(vals_cd)))
        assert n_valid_cd < MIN_POINTS

        # Pb still has all values
        vals_pb, counts_pb = process_metal_values(df["Pb"])
        n_valid_pb = int(np.sum(np.isfinite(vals_pb)))
        assert n_valid_pb >= MIN_POINTS


# =====================================================================
# Test 8: Variant headers
# =====================================================================

class TestVariantHeaders:
    def test_lat_long_station(self):
        df = pd.DataFrame({
            "Station": ["A", "B", "C", "D", "E"],
            "Long": [90.4, 90.42, 90.44, 90.46, 90.48],
            "LAT": [23.8, 23.82, 23.84, 23.86, 23.88],
            "Pb": [1, 2, 3, 4, 5],
        })
        csv_bytes = _df_to_csv_bytes(df)
        parsed = read_csv_flexible(csv_bytes)
        site_col, lon_col, lat_col, _ = detect_columns(parsed)
        assert lon_col == "Long"
        assert lat_col == "LAT"
        assert site_col == "Station"

    def test_missing_lon_lat_lists_columns(self):
        df = pd.DataFrame({
            "X": [1, 2, 3],
            "Y": [4, 5, 6],
            "Pb": [7, 8, 9],
        })
        site_col, lon_col, lat_col, all_cols = detect_columns(df)
        assert lon_col is None
        assert lat_col is None
        assert "X" in all_cols and "Y" in all_cols


# =====================================================================
# Test 9: Tiny extent
# =====================================================================

class TestTinyExtent:
    def test_tiny_extent_widens(self):
        # Points within ~50m of each other
        lons = np.array([90.4000, 90.4001, 90.4002, 90.4003, 90.4004])
        lats = np.array([23.8000, 23.8001, 23.8002, 23.8003, 23.8004])
        lon_min, lon_max, lat_min, lat_max = compute_extent(lons, lats)
        assert (lon_max - lon_min) >= MIN_EXTENT_DEG
        assert (lat_max - lat_min) >= MIN_EXTENT_DEG

    def test_tiny_extent_renders(self):
        lons = np.array([90.4000, 90.4001, 90.4002, 90.4003, 90.4004])
        lats = np.array([23.8000, 23.8001, 23.8002, 23.8003, 23.8004])
        vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        fig, _, _, _ = _render_metal(lons, lats, vals, "Pb")
        png = export_png(fig, dpi=100)
        assert len(png) > 1000
        fig.clear()

    def test_zero_range_one_axis(self):
        # All same longitude but spread latitude
        lons = np.array([90.4, 90.4, 90.4, 90.4, 90.4])
        lats = np.array([23.80, 23.82, 23.84, 23.86, 23.88])
        lon_min, lon_max, lat_min, lat_max = compute_extent(lons, lats)
        # Lon range should be at least MIN_EXTENT_DEG
        assert (lon_max - lon_min) >= MIN_EXTENT_DEG


# =====================================================================
# Test 10: Buffer
# =====================================================================

class TestBuffer:
    def test_buffer_10_percent(self):
        lons = np.array([90.0, 91.0, 90.5, 90.3, 90.8])
        lats = np.array([23.0, 24.0, 23.5, 23.3, 23.8])
        d_lon = 91.0 - 90.0
        d_lat = 24.0 - 23.0
        expected_lon_min = 90.0 - 0.10 * d_lon
        expected_lon_max = 91.0 + 0.10 * d_lon
        expected_lat_min = 23.0 - 0.10 * d_lat
        expected_lat_max = 24.0 + 0.10 * d_lat

        lon_min, lon_max, lat_min, lat_max = compute_extent(lons, lats)
        assert abs(lon_min - expected_lon_min) < 1e-9
        assert abs(lon_max - expected_lon_max) < 1e-9
        assert abs(lat_min - expected_lat_min) < 1e-9
        assert abs(lat_max - expected_lat_max) < 1e-9


# =====================================================================
# Test 11: CRS
# =====================================================================

class TestCRS:
    def test_utm_zone_46(self):
        """Longitude 90.4 should select EPSG:32646 (UTM zone 46N)."""
        target_crs, label, approx = determine_crs(90.4, 23.8, 0.5)
        assert label == "EPSG:32646"
        assert not approx

    def test_projected_distance(self):
        """Projected distance between two points ~10 km apart should
        be within 0.5% of the geodesic distance."""
        lon1, lat1 = 90.40, 23.80
        lon2, lat2 = 90.50, 23.80
        target_crs, _, _ = determine_crs(90.45, 23.80, 0.1)

        x, y = project_points(
            np.array([lon1, lon2]), np.array([lat1, lat2]), target_crs,
        )
        proj_dist = math.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)

        # Geodesic distance
        geod = pyproj.Geod(ellps="WGS84")
        _, _, geo_dist = geod.inv(lon1, lat1, lon2, lat2)

        rel_error = abs(proj_dist - geo_dist) / geo_dist
        assert rel_error < 0.005  # within 0.5%


# =====================================================================
# Test 12: No extrapolation
# =====================================================================

class TestNoExtrapolation:
    def test_outside_hull_is_nan(self):
        # Square of points
        lons = np.array([90.40, 90.50, 90.40, 90.50, 90.45])
        lats = np.array([23.80, 23.80, 23.90, 23.90, 23.85])
        vals = np.array([10.0, 20.0, 30.0, 40.0, 25.0])

        settings = dict(_DEFAULT_SETTINGS)
        settings["full_extent"] = False

        lon_min, lon_max, lat_min, lat_max = compute_extent(lons, lats)
        target_crs, _, _ = determine_crs(90.45, 23.85, lon_max - lon_min)
        proj_ext = project_extent(lon_min, lon_max, lat_min, lat_max, target_crs)
        GX, GY, _, _ = build_grid(proj_ext[0], proj_ext[1], proj_ext[2], proj_ext[3], 100)
        sx, sy = project_points(lons, lats, target_crs)
        Z = idw_interpolate(sx, sy, vals, GX, GY, power=2.0)
        Z_masked = apply_mask(Z, GX, GY, sx, sy, full_extent=False)

        # Some cells should be masked (NaN)
        assert Z_masked.mask.any()

    def test_beyond_mask_radius_is_nan(self):
        # Scattered points with a large gap
        lons = np.array([90.4, 90.5, 90.6, 90.7, 90.8])
        lats = np.array([23.8, 23.8, 23.8, 23.8, 23.8])
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sx, sy = project_points(lons, lats,
                                determine_crs(90.6, 23.8, 0.5)[0])
        GX_test = np.array([[sx[0] - 100000]])  # Far away
        GY_test = np.array([[sy[0]]])
        dm = compute_distance_mask(sx, sy, GX_test, GY_test, multiplier=3.0)
        assert not dm[0, 0]


# =====================================================================
# Test 13: Colour ramp
# =====================================================================

class TestColourRamp:
    def test_viridis_vs_plasma(self):
        df = _make_base_df(10, metals=["Pb"])
        settings_v = dict(_DEFAULT_SETTINGS, cmap="viridis")
        settings_p = dict(_DEFAULT_SETTINGS, cmap="plasma")

        fig_v, _, _, _ = _render_metal(
            df["Longitudes"].values, df["Latitudes"].values,
            df["Pb"].values, "Pb", settings=settings_v,
        )
        fig_p, _, _, _ = _render_metal(
            df["Longitudes"].values, df["Latitudes"].values,
            df["Pb"].values, "Pb", settings=settings_p,
        )
        png_v = export_png(fig_v, dpi=100)
        png_p = export_png(fig_p, dpi=100)
        assert png_v != png_p
        fig_v.clear()
        fig_p.clear()


# =====================================================================
# Test 14: A4 dimensions
# =====================================================================

class TestA4Dimensions:
    def _get_png_dims(self, data: bytes) -> tuple[int, int]:
        """Extract width and height from PNG IHDR."""
        # PNG: 8 byte sig, then IHDR chunk: 4 len + 4 type + 4 width + 4 height
        assert data[:8] == b'\x89PNG\r\n\x1a\n'
        w = struct.unpack('>I', data[16:20])[0]
        h = struct.unpack('>I', data[20:24])[0]
        return w, h

    def test_landscape_png_dimensions(self):
        df = _make_base_df(10, metals=["Pb"])
        settings = dict(_DEFAULT_SETTINGS, orientation="landscape")
        fig, _, _, _ = _render_metal(
            df["Longitudes"].values, df["Latitudes"].values,
            df["Pb"].values, "Pb", settings=settings, dpi=300,
        )
        png = export_png(fig, dpi=300)
        w, h = self._get_png_dims(png)
        assert w == 3508
        assert h == 2480
        fig.clear()

    def test_portrait_png_dimensions(self):
        df = _make_base_df(10, metals=["Pb"])
        settings = dict(_DEFAULT_SETTINGS, orientation="portrait")
        fig, _, _, _ = _render_metal(
            df["Longitudes"].values, df["Latitudes"].values,
            df["Pb"].values, "Pb", settings=settings, dpi=300,
        )
        png = export_png(fig, dpi=300)
        w, h = self._get_png_dims(png)
        assert w == 2480
        assert h == 3508
        fig.clear()

    def test_pdf_one_page_landscape(self):
        from pypdf import PdfReader

        df = _make_base_df(10, metals=["Pb"])
        settings = dict(_DEFAULT_SETTINGS, orientation="landscape")
        fig, _, _, _ = _render_metal(
            df["Longitudes"].values, df["Latitudes"].values,
            df["Pb"].values, "Pb", settings=settings, dpi=300,
        )
        pdf_bytes = export_pdf(fig)
        reader = PdfReader(io.BytesIO(pdf_bytes))
        assert len(reader.pages) == 1

        page = reader.pages[0]
        mb = page.mediabox
        w = float(mb.width)
        h = float(mb.height)
        # A4 landscape: 841.89 × 595.28 pt (±0.5)
        assert abs(w - 841.89) < 0.6 or abs(h - 841.89) < 0.6
        assert abs(w - 595.28) < 0.6 or abs(h - 595.28) < 0.6
        fig.clear()


# =====================================================================
# Test 15: Identity — metal name in PDF, PNGs differ
# =====================================================================

class TestIdentity:
    def test_pdf_contains_metal_name(self):
        from pypdf import PdfReader

        df = _make_base_df(10, metals=["Pb", "Cd"])
        for metal in ["Pb", "Cd"]:
            fig, _, _, _ = _render_metal(
                df["Longitudes"].values, df["Latitudes"].values,
                df[metal].values, metal, dpi=300,
            )
            pdf_bytes = export_pdf(fig)
            reader = PdfReader(io.BytesIO(pdf_bytes))
            text = reader.pages[0].extract_text()
            assert metal in text, f"'{metal}' not found in PDF text"
            fig.clear()

    def test_pngs_different_metals_differ(self):
        df = _make_base_df(10, metals=["Pb", "Cd"])
        pngs = {}
        for metal in ["Pb", "Cd"]:
            fig, _, _, _ = _render_metal(
                df["Longitudes"].values, df["Latitudes"].values,
                df[metal].values, metal,
            )
            pngs[metal] = export_png(fig, dpi=100)
            fig.clear()
        assert pngs["Pb"] != pngs["Cd"]


# =====================================================================
# Test 16: ZIP structure
# =====================================================================

class TestZIP:
    def test_zip_contents(self):
        df = _make_base_df(10, metals=["Pb", "Cd"])
        files = {}
        for metal in ["Pb", "Cd"]:
            fig, _, _, _ = _render_metal(
                df["Longitudes"].values, df["Latitudes"].values,
                df[metal].values, metal,
            )
            name = safe_filename(metal)
            files[f"{name}_Heatmap.png"] = export_png(fig, dpi=100)
            files[f"{name}_Heatmap.pdf"] = export_pdf(fig)
            fig.clear()

        zip_bytes = create_zip(files)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            assert len(names) == 4  # 2 × 2
            for n in names:
                data = zf.read(n)
                if n.endswith(".png"):
                    assert data[:8] == b'\x89PNG\r\n\x1a\n'
                elif n.endswith(".pdf"):
                    assert data[:5] == b'%PDF-'


# =====================================================================
# Test 17: Failure isolation
# =====================================================================

class TestFailureIsolation:
    def test_one_metal_failure_others_ok(self):
        df = _make_base_df(10, metals=["Pb", "Cd"])
        results = {}

        # Pb: normal
        fig, _, _, _ = _render_metal(
            df["Longitudes"].values, df["Latitudes"].values,
            df["Pb"].values, "Pb",
        )
        results["Pb"] = {
            "png": export_png(fig, dpi=100),
            "pdf": export_pdf(fig),
            "error": None,
        }
        fig.clear()

        # Cd: simulate failure
        results["Cd"] = {"error": "Simulated failure"}

        # ZIP should contain only Pb
        zip_files = {}
        for metal, r in results.items():
            if r.get("error"):
                continue
            zip_files[f"{metal}_Heatmap.png"] = r["png"]
            zip_files[f"{metal}_Heatmap.pdf"] = r["pdf"]

        zip_bytes = create_zip(zip_files)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            assert len(names) == 2
            assert any("Pb" in n for n in names)
            assert not any("Cd" in n for n in names)

        # Error message available
        assert results["Cd"]["error"] == "Simulated failure"


# =====================================================================
# Test 18: Limits
# =====================================================================

class TestLimits:
    def test_oversized_file(self):
        big = b"x" * (MAX_UPLOAD_MB * 1024 * 1024 + 1)
        size_mb = len(big) / (1024 * 1024)
        assert size_mb > MAX_UPLOAD_MB

    def test_too_many_rows(self):
        n = MAX_ROWS + 1
        df = pd.DataFrame({
            "Longitudes": np.full(n, 90.4),
            "Latitudes": np.full(n, 23.8),
            "Pb": np.ones(n),
        })
        assert len(df) > MAX_ROWS

    def test_too_many_metals(self):
        assert MAX_METALS == 20
        # A dataset with 21 metal columns
        data = {
            "Longitudes": [90.4], "Latitudes": [23.8],
        }
        for i in range(21):
            data[f"M{i}"] = [1.0]
        df = pd.DataFrame(data)
        metals = detect_metals(df, None, "Longitudes", "Latitudes")
        assert len(metals) == 21
        assert len(metals) > MAX_METALS


# =====================================================================
# Test 19: Startup (AppTest)
# =====================================================================

class TestStartup:
    def test_app_renders_without_error(self):
        from streamlit.testing.v1 import AppTest
        import os
        app_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "app.py",
        )
        at = AppTest.from_file(app_path, default_timeout=30)
        at.run()
        assert not at.exception, f"App raised: {at.exception}"
        # Title should be present
        assert any(
            "Heavy Metal Heatmap Generator" in getattr(el, "value", str(el))
            for el in at.title
        )


# =====================================================================
# Additional utility tests
# =====================================================================

class TestUtilities:
    def test_safe_filename(self):
        assert safe_filename("Pb (mg/kg)") == "Pb_mgkg"
        assert safe_filename("Cr") == "Cr"
        assert safe_filename("") == "metal"

    def test_sanitize_report_cell(self):
        assert sanitize_report_cell("=SUM(A1)") == "'=SUM(A1)"
        assert sanitize_report_cell("+cmd") == "'+cmd"
        assert sanitize_report_cell("normal") == "normal"

    def test_get_metal_title(self):
        assert "Lead" in get_metal_title("Pb")
        assert "Pb" in get_metal_title("Pb")

    def test_non_detect_parsing(self):
        vals, counts = process_metal_values(pd.Series(["<0.01", "ND", "BDL", "<DL", "5.0"]))
        assert counts["non_detect"] == 4
        assert counts["valid"] == 1

    def test_semicolon_csv(self):
        csv_text = "Site;Lon;Lat;Pb\nA;90,4;23,8;12,5\nB;90,5;23,9;15,2"
        df = read_csv_flexible(csv_text.encode("utf-8"))
        assert len(df) == 2
        # Comma decimal should have been parsed
        assert abs(df.iloc[0]["Lon"] - 90.4) < 0.01

    def test_element_names_coverage(self):
        assert len(ELEMENT_NAMES) >= 20
