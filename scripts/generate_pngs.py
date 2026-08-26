import sys
import cfgrib
import pandas as pd
import os
from zoneinfo import ZoneInfo
from scipy.spatial.distance import cdist
import numpy as np
import gc
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
import matplotlib.colors as mcolors
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from scipy.spatial import Delaunay
from PIL import Image
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# ------------------------------
# Eingabe-/Ausgabe
# ------------------------------
data_dir = sys.argv[1]        # z.B. "output"
output_dir = sys.argv[2]      # z.B. "output/maps"
var_type = sys.argv[3].lower()# 't2m', 'tp', 'ww', 'cape_ml', 'dbz_cmax', 'wind', etc.
grid_dir = sys.argv[4] if len(sys.argv) > 4 else "data/grid"
cape_dir = sys.argv[5] if len(sys.argv) > 5 else None
wshearu_dir = sys.argv[6] if len(sys.argv) > 6 else None
wshearv_dir = sys.argv[7] if len(sys.argv) > 7 else None

os.makedirs(output_dir, exist_ok=True)

# ICON-RUC läuft auf dem nativen Dreiecksgitter - die Gitterkoordinaten
# müssen separat aus den von DWD bereitgestellten CLAT-/CLON-GRIB2-Dateien
# geladen werden.
clat_path = os.path.join(grid_dir, "clat.grib2")
clon_path = os.path.join(grid_dir, "clon.grib2")

for gp in (clat_path, clon_path):
    if not os.path.exists(gp):
        raise FileNotFoundError(f"Grid-Datei nicht gefunden: {gp}")

# ------------------------------
# WW-Farben
# ------------------------------
ww_colors_base = {
    0: "#FFFFFF", 1: "#D3D3D3", 2: "#A9A9A9", 3: "#696969",
    45: "#FFFF00", 48: "#FFD700",
    56: "#FFA500", 57: "#C06A00",
    51: "#00FF00", 53: "#00C300", 55: "#009700",
    61: "#00FF00", 63: "#00C300", 65: "#009700",
    80: "#00FF00", 81: "#00C300", 82: "#009700",
    66: "#FF6347", 67: "#8B0000",
    71: "#ADD8E6", 73: "#6495ED", 75: "#00008B",
    77: "#ADD8E6", 85: "#6495ED", 86: "#00008B",
    95: "#FF77FF", 96: "#C71585", 99: "#C71585"
}

ignore_codes = {4}

# ------------------------------
# Temperatur-Farben
# ------------------------------
t2m_bounds = list(range(-36, 50, 2))
t2m_colors = LinearSegmentedColormap.from_list(
    "t2m_smoooth",
    [
        "#F675F4", "#F428E9", "#B117B5", "#950CA2", "#640180",
        "#3E007F", "#00337E", "#005295", "#1292FF", "#49ACFF",
        "#8FCDFF", "#B4DBFF", "#B9ECDD", "#88D4AD", "#07A125",
        "#3FC107", "#9DE004", "#E7F700", "#F3CD0A", "#EE5505",
        "#C81904", "#AF0E14", "#620001", "#C87879", "#FACACA",
        "#E1E1E1", "#6D6D6D"
    ],
    N=len(t2m_bounds)
)
t2m_norm = BoundaryNorm(t2m_bounds, ncolors=len(t2m_bounds))

# ------------------------------
# Niederschlags-Farben (tp)
# ------------------------------
prec_bounds = [0.0, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
               12, 14, 16, 20, 24, 30, 40, 50, 60, 80, 100, 125]
prec_colors = ListedColormap([
    "#FFFFFF", "#B4D7FF", "#75BAFF", "#349AFF", "#0582FF", "#0069D2",
    "#003680", "#148F1B", "#1ACF06", "#64ED07", "#FFF32B",
    "#E9DC01", "#F06000", "#FF7F26", "#FFA66A", "#F94E78",
    "#F71E53", "#BE0000", "#880000", "#64007F", "#C201FC",
    "#DD66FE", "#EBA6FF", "#F9E7FF", "#D4D4D4"
])
prec_norm = mcolors.BoundaryNorm(prec_bounds, prec_colors.N)

# ------------------------------
# Aufsummierter Niederschlag (tp_acc)
# ------------------------------
tp_acc_bounds = [0.0, 0.1, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100,
                  125, 150, 175, 200, 250, 300, 400, 500]
tp_acc_colors = ListedColormap([
    "#FFFFFF", "#B4D7FF", "#75BAFF", "#349AFF", "#0582FF", "#0069D2",
    "#003680", "#148F1B", "#1ACF06", "#64ED07", "#FFF32B",
    "#E9DC01", "#F06000", "#FF7F26", "#FFA66A", "#F94E78",
    "#F71E53", "#BE0000", "#880000", "#64007F", "#C201FC",
    "#DD66FE", "#EBA6FF", "#F9E7FF", "#D4D4D4", "#969696"
])
tp_acc_norm = mcolors.BoundaryNorm(tp_acc_bounds, tp_acc_colors.N)

# ------------------------------
# CAPE-Farben
# ------------------------------
cape_bounds = [0, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000, 1500, 2000, 2500, 3000]
cape_colors = ListedColormap([
    "#676767", "#006400", "#008000", "#00CC00", "#66FF00", "#FFFF00",
    "#FFCC00", "#FF9900", "#FF6600", "#FF3300", "#FF0000", "#FF0095",
    "#FC439F", "#FF88D3", "#FF99FF"
])
cape_norm = mcolors.BoundaryNorm(cape_bounds, cape_colors.N)

# ------------------------------
# DBZ-CMAX Farben
# ------------------------------
dbz_bounds = [0, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 63, 67, 70]
dbz_colors = ListedColormap([
    "#676767", "#FFFFFF", "#B3EFED", "#8CE7E2", "#00F5ED",
    "#00CEF0", "#01AFF4", "#028DF6", "#014FF7", "#0000F6",
    "#00FF01", "#01DF00", "#00D000", "#00BF00", "#00A701",
    "#019700", "#FFFF00", "#F9F000", "#EDD200", "#E7B500",
    "#FF5000", "#FF2801", "#F40000", "#EA0001", "#CC0000",
    "#FFC8FF", "#E9A1EA", "#D379D3", "#BE55BE", "#960E96"
])
dbz_colors.set_under(alpha=0)
dbz_norm = mcolors.BoundaryNorm(dbz_bounds, dbz_colors.N)

# ------------------------------
# Windböen-Farben
# ------------------------------
wind_bounds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 200, 220, 240, 260, 280, 300]
wind_colors = ListedColormap([
    "#68AD05", "#8DC00B", "#B1D415", "#D5E81C", "#FBFC22",
    "#FAD024", "#F9A427", "#FC7929", "#FB4D2B", "#EA2B57",
    "#FB22A5", "#FC22CE", "#FC22F5", "#FC62F8", "#FD80F8",
    "#FFBFFC", "#FEDFFE", "#FEFFFF", "#E1E0FF", "#C3C3FF",
    "#A5A5FF", "#A5A5FF", "#6868FE"
])
wind_norm = mcolors.BoundaryNorm(wind_bounds, wind_colors.N)

# ------------------------------
# Schneehöhen-Farben
# ------------------------------
snow_bounds = [0, 0.1, 0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200, 250, 300, 400]
snow_colors = ListedColormap([
    "#F8F8F8", "#DCDBFA", "#AAA9C8", "#75BAFF", "#349AFF", "#0582FF",
    "#0069D2", "#004F9C", "#01327F", "#4B007F", "#64007F", "#9101BB",
    "#C300FC", "#D235FF", "#EBA6FF", "#F4CEFF", "#FAB2CA", "#FF9798",
    "#FE6E6E", "#DF093F", "#BE0000", "#A40000", "#880000", "#460000"
])
snow_norm = mcolors.BoundaryNorm(snow_bounds, snow_colors.N)

# ------------------------------
# SRH-Farben
# ------------------------------
srh_bounds = [-250, -200, -150, -100, -50, -25, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 1000, 1250, 1500]
srh_colors = ListedColormap([
    "#0069D2", "#0482FF", "#359AFF", "#75BAFF", "#D2E9FF", "#FFFFFF",
    "#B4FF5A", "#63ED07", "#1ACF05", "#97C90E", "#E8DC00", "#FFF42B",
    "#FFA66A", "#F84E78", "#F71E54", "#BF0000", "#880000", "#64007F",
    "#C200FB", "#DD66FF", "#EBA6FF", "#EFC7FA"
])
srh_norm = mcolors.BoundaryNorm(srh_bounds, srh_colors.N)

# ------------------------------
# EHI-Farben
# ------------------------------
ehi_bounds = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
ehi_colors = ListedColormap([
    "#75BAFF", "#0482FF", "#1ACF05", "#63ED07",
    "#FFF42B", "#E8DC00", "#FF7F27", "#F71E54", "#880000",
    "#64007F", "#C200FB", "#DD66FF", "#EBA6FF", "#B97A57"
])
ehi_colors.set_under(alpha=0)
ehi_norm = mcolors.BoundaryNorm(ehi_bounds, ehi_colors.N)

# ------------------------------
# SCP-Farben
# ------------------------------
scp_bounds = [0, 0.2, 0.5, 1, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50]
scp_colors = ListedColormap([
    "#FFFFFF", "#D2E9FF", "#75BAFF", "#0069D2", "#148F1B",
    "#63ED07", "#FFF42B", "#E8DC00", "#FF7F27", "#F71E54",
    "#880000", "#64007F", "#C200FB", "#DD66FF", "#EBA6FF",
    "#B97A57"
])
scp_norm = mcolors.BoundaryNorm(scp_bounds, scp_colors.N)

# Bounding Box ICON-RUC (aus clat/clon ermittelt)
extent = [-4.1616, 20.5444, 43.0440, 58.1647]  # lon_min, lon_max, lat_min, lat_max

FOOTER_TEXTS = {
    "ww": "Signifikantes Wetter",
    "t2m": "Temperatur 2m (°C)",
    "tp": "Niederschlag, 1Std (mm)",
    "tp_acc": "Akkumulierter Niederschlag (mm)",
    "cape_ml": "CAPE-Index (J/kg)",
    "dbz_cmax": "Sim. max. Radarreflektivität (dBZ)",
    "wind": "Windböen (km/h)",
    "snow": "Schneehöhe (cm)",
    "srh3km": "Helizität 0-3km (m²/s²)",
    "ehi": "Energy Helicity Index (EHI)",
    "scp": "Supercell Composite Parameter (SCP)",
}

VALUE_UNITS = {
    "ww": "",
    "t2m": "°C",
    "tp": "mm",
    "tp_acc": "mm",
    "cape_ml": "J/kg",
    "dbz_cmax": "dBZ",
    "wind": "km/h",
    "snow": "cm",
    "srh3km": "m²/s²",
    "ehi": "",
    "scp": "",
}

VALUE_DECIMALS = {
    "ww": 0,
    "t2m": 1,
    "tp": 1,
    "tp_acc": 1,
    "cape_ml": 0,
    "dbz_cmax": 0,
    "wind": 0,
    "snow": 1,
    "srh3km": 0,
    "ehi": 2,
    "scp": 2,
}

VALUE_NODATA = -9999.0

# ------------------------------
# EPSG:4326 -> EPSG:3857 (Web Mercator)
# ------------------------------
EARTH_RADIUS = 6378137.0
WEBMERCATOR_WIDTH = 1024

def lonlat_to_webmercator(lon_deg, lat_deg):
    x = EARTH_RADIUS * np.radians(lon_deg)
    y = EARTH_RADIUS * np.log(np.tan(np.pi / 4 + np.radians(lat_deg) / 2))
    return x, y

def webmercator_target_grid(extent, out_width=WEBMERCATOR_WIDTH):
    lon_min, lon_max, lat_min, lat_max = extent
    x_min, y_min = lonlat_to_webmercator(lon_min, lat_min)
    x_max, y_max = lonlat_to_webmercator(lon_max, lat_max)
    aspect = (y_max - y_min) / (x_max - x_min)
    out_height = max(int(round(out_width * aspect)), 1)
    x_new = np.linspace(x_min, x_max, out_width)
    y_new = np.linspace(y_min, y_max, out_height)
    return x_new, y_new

# Domain Extent in Web Mercator berechnen (für Manifest)
_dom_x_min, _dom_y_min = lonlat_to_webmercator(extent[0], extent[2])
_dom_x_max, _dom_y_max = lonlat_to_webmercator(extent[1], extent[3])
DOMAIN_EXTENT_3857 = [float(_dom_x_min), float(_dom_y_min), float(_dom_x_max), float(_dom_y_max)]

def data_to_rgba(data, cmap, norm):
    """Wandelt 2D-Datenarray in RGBA-uint8-Array um."""
    rgba = cmap(norm(data))
    rgba = (rgba * 255).astype(np.uint8)
    nan_mask = ~np.isfinite(data)
    rgba[nan_mask, 3] = 0
    return rgba

def save_transparent_webp(data, cmap, norm, out_path):
    rgba = data_to_rgba(data, cmap, norm)
    img = Image.fromarray(rgba[::-1, :, :], mode="RGBA")
    img.save(out_path, format="WEBP", lossless=True, method=4)

def _load_grid_coords(path):
    """Lädt Gitterkoordinaten aus CLAT-/CLON-GRIB2-Datei."""
    ds_grid = cfgrib.open_dataset(path)
    varname = list(ds_grid.data_vars)[0]
    values = np.asarray(ds_grid[varname].values).ravel()
    ds_grid.close()

    if np.nanmax(np.abs(values)) <= (np.pi + 0.1):
        values = np.rad2deg(values)
    return values

lats = _load_grid_coords(clat_path)
lons = _load_grid_coords(clon_path)

lons_merc = EARTH_RADIUS * np.radians(lons)
lats_merc = EARTH_RADIUS * np.log(np.tan(np.pi / 4 + np.radians(lats) / 2))
points_merc_base = np.column_stack((lons_merc, lats_merc))

# --- Sanity check the grid before triangulating ---
finite_mask = np.all(np.isfinite(points_merc_base), axis=1)
n_bad = (~finite_mask).sum()
if n_bad:
    print(f"Warnung: {n_bad} nicht-endliche Gitterpunkte gefunden, werden entfernt.")

if not np.all(finite_mask):
    # keep a mapping so later code (which indexes by original grid point order)
    # still works — easiest is to filter once, globally, and use the same
    # filtered index everywhere `render_data` is built from the raw values.
    valid_idx = np.nonzero(finite_mask)[0]
    points_merc_base = points_merc_base[valid_idx]
else:
    valid_idx = None

print(f"Gitterpunkte: {points_merc_base.shape[0]} (finite)")
print("Baue Basis-Triangulation für Interpolation & Hüllen-Maske auf ...")
base_tri = Delaunay(points_merc_base, qhull_options="QJ")
interp_linear_base = LinearNDInterpolator(base_tri, np.zeros(len(points_merc_base), dtype=np.float64))

# Zielgitter (Web Mercator) ist für alle Dateien identisch -> einmal berechnen
x_new, y_new = webmercator_target_grid(extent, out_width=WEBMERCATOR_WIDTH)
xx, yy = np.meshgrid(x_new, y_new)
target_points = np.column_stack((xx.ravel(), yy.ravel()))

# Huellen-Maske einmalig berechnen: Punkte ausserhalb der Dreiecksgitter-Huelle -> NaN
outside_hull = base_tri.find_simplex(target_points) < 0
outside_hull_2d = outside_hull.reshape(xx.shape)

# ------------------------------
# Dateien durchgehen
# ------------------------------
all_files_global = sorted([f for f in os.listdir(data_dir) if f.endswith(".grib2")])

for filename in all_files_global:
    path = os.path.join(data_dir, filename)
    ds = cfgrib.open_dataset(path)

    # --------------------------
    # Daten je Typ
    # --------------------------
    if var_type == "t2m":
        if "t2m" not in ds:
            print(f"Keine t2m in {filename}")
            ds.close()
            continue
        data = ds["t2m"].values - 273.15
        cmap, norm = t2m_colors, t2m_norm
    elif var_type == "tp":
        if "tprate" not in ds:
            print(f"Keine tprate in {filename}")
            ds.close()
            continue
        data = ds["tprate"].values
        data[data < 0.1] = np.nan
        cmap, norm = prec_colors, prec_norm
    elif var_type == "tp_acc":
        if "tp" not in ds:
            print(f"Keine tp in {filename}")
            ds.close()
            continue
        data = ds["tp"].values
        data[data < 0.1] = np.nan
        cmap, norm = tp_acc_colors, tp_acc_norm
    elif var_type == "ww":
        varname = next((vn for vn in ds.data_vars if vn in ["WW", "weather"]), None)
        if varname is None:
            print(f"Keine WW in {filename}")
            ds.close()
            continue
        data = ds[varname].values
        cmap, norm = None, None
    elif var_type == "cape_ml":
        if "CAPE_ML" not in ds:
            print(f"Keine CAPE_ML in {filename}")
            ds.close()
            continue
        data = ds["CAPE_ML"].values
        data[data < 0] = np.nan
        cmap, norm = cape_colors, cape_norm
    elif var_type == "dbz_cmax":
        if "DBZ_CMAX" not in ds:
            print(f"Keine DBZ_CMAX in {filename}")
            ds.close()
            continue
        data = ds["DBZ_CMAX"].values
        cmap, norm = dbz_colors, dbz_norm
    elif var_type == "wind":
        if "fg10" not in ds:
            print(f"Keine fg10 in {filename}")
            ds.close()
            continue
        data = ds["fg10"].values * 3.6  # m/s -> km/h
        data[data < 0] = np.nan
        cmap, norm = wind_colors, wind_norm
    elif var_type == "snow":
        if "sde" not in ds:
            print(f"Keine sde in {filename}")
            ds.close()
            continue
        data = ds["sde"].values * 100  # -> cm
        data[data < 0] = np.nan
        cmap, norm = snow_colors, snow_norm
    elif var_type == "srh3km":
        if "hlcy" not in ds:
            print(f"Keine hlcy in {filename}")
            ds.close()
            continue
        data = ds["hlcy"].values
        cmap, norm = srh_colors, srh_norm
    elif var_type == "ehi":
        if cape_dir is None:
            print("Kein cape_dir angegeben")
            ds.close()
            continue
        suffix = filename.replace("srh3km_", "")
        cape_filename = f"cape_ml_{suffix}"
        cape_path = os.path.join(cape_dir, cape_filename)
        if not os.path.exists(cape_path):
            print(f"Kein CAPE-File: {cape_path}")
            ds.close()
            continue
        ds_cape = cfgrib.open_dataset(cape_path)
        if "hlcy" not in ds or "CAPE_ML" not in ds_cape:
            print(f"Variablen fehlen")
            ds.close()
            ds_cape.close()
            continue
        srh = ds["hlcy"].values
        cape = ds_cape["CAPE_ML"].values
        cape[cape < 0] = 0
        data = (cape * srh) / 160000.0
        data[data < 0] = np.nan
        cmap, norm = ehi_colors, ehi_norm
        ds_cape.close()
    elif var_type == "scp":
        if None in [cape_dir, wshearu_dir, wshearv_dir]:
            print("SCP benötigt cape_dir, wshearu_dir, wshearv_dir")
            ds.close()
            continue
        suffix = filename.replace("srh3km_", "")
        cape_path = os.path.join(cape_dir, f"cape_ml_{suffix}")
        wshearu_path = os.path.join(wshearu_dir, f"wshear_u_{suffix}")
        wshearv_path = os.path.join(wshearv_dir, f"wshear_v_{suffix}")

        missing = False
        for p in [cape_path, wshearu_path, wshearv_path]:
            if not os.path.exists(p):
                print(f"Datei nicht gefunden: {p}")
                missing = True
        if missing:
            ds.close()
            continue

        ds_cape = cfgrib.open_dataset(cape_path)
        ds_wshearu = cfgrib.open_dataset(wshearu_path)
        ds_wshearv = cfgrib.open_dataset(wshearv_path)

        if "hlcy" not in ds:
            print(f"hlcy fehlt in {filename}")
            ds.close()
            ds_cape.close()
            ds_wshearu.close()
            ds_wshearv.close()
            continue

        srh = ds["hlcy"].values
        cape = ds_cape["CAPE_ML"].values
        bs06 = np.sqrt(ds_wshearu["WSHEAR_U"].values**2 + ds_wshearv["WSHEAR_V"].values**2)

        cape[cape < 0] = 0
        cape_term = cape / 1000
        srh_term = np.maximum(0, srh) / 50
        ebs_term = np.where(bs06 < 10, 0.0, np.where(bs06 > 20, 1.0, bs06 / 20))

        data = np.round(cape_term * srh_term * ebs_term, 2)
        data[data < 0] = np.nan
        cmap, norm = scp_colors, scp_norm

        ds_cape.close()
        ds_wshearu.close()
        ds_wshearv.close()
    else:
        print(f"Unbekannter var_type: {var_type}")
        ds.close()
        continue

    if data.ndim == 3:
        data = data[0]

    # --------------------------
    # Zeiten
    # --------------------------
    run_time_utc = pd.to_datetime(ds["time"].values) if "time" in ds else None
    if "valid_time" in ds:
        valid_time_raw = ds["valid_time"].values
        valid_time_utc = pd.to_datetime(valid_time_raw[0]) if np.ndim(valid_time_raw) > 0 else pd.to_datetime(valid_time_raw)
    else:
        step = pd.to_timedelta(ds["step"].values[0]) if "step" in ds else pd.to_timedelta(0)
        valid_time_utc = run_time_utc + step if run_time_utc else None

    valid_time_local = valid_time_utc.tz_localize("UTC").astimezone(ZoneInfo("Europe/Berlin")) if valid_time_utc else None

    ds.close()

    # --------------------------
    # Farb-Mapping je Typ (auf Dreiecksgitter-Daten)
    # --------------------------
    if var_type == "ww":
        valid_mask = np.isfinite(data)  # nur numerisch ungültige Rohdaten raus, nicht "kein Wetter"
        codes = np.unique(data[valid_mask]).astype(int)
        codes = [c for c in codes if c in ww_colors_base and c not in ignore_codes]
        codes.sort()
        cmap = ListedColormap([ww_colors_base[c] for c in codes]) if codes else ListedColormap(["#FFFFFF00"])
        norm = mcolors.Normalize(vmin=-0.5, vmax=max(len(codes) - 0.5, 0.5))
        code2idx = {c: i for i, c in enumerate(codes)}

        # WICHTIG: "kein Wetter" bekommt einen eigenen Index (-1), keine NaN
        idx_data = np.full_like(data, fill_value=-1, dtype=float)
        for c, i in code2idx.items():
            idx_data[data == c] = i
        render_data = idx_data
    else:
        render_data = data.copy()

    # --------------------------
    # Interpolation auf Web-Mercator-Zielgitter
    # --------------------------
    if render_data.shape[0] != points_merc_base.shape[0]:
        print(f"{filename}: Anzahl Datenpunkte passt nicht zum Grid, überspringe")
        continue

    if var_type == "ww":
        valid_mask = np.isfinite(render_data)
        if not np.any(valid_mask):
            print(f"{filename}: Keine gültigen Daten")
            continue
        interpolator_nn = NearestNDInterpolator(points_merc_base, render_data)  # ALLE Punkte, kein valid_mask-Filter
        render_data_merc = interpolator_nn(target_points).reshape(xx.shape)
        render_data_merc[outside_hull_2d] = np.nan
        render_data_merc[render_data_merc < 0] = np.nan  # -1 ("kein Wetter") am Ende zu NaN/transparent machen
    else:
        # Kontinuierliche Felder: wiederverwendete Triangulation, nur Werte tauschen.
        # Ausserhalb der Huelle liefert LinearNDInterpolator ohnehin automatisch NaN.
        interp_linear_base.values[:, 0] = render_data.astype(np.float64)
        render_data_merc = interp_linear_base(target_points).reshape(xx.shape)

    # Huellen-Maske anwenden (für ww notwendig, für linear redundant aber unschädlich)
    render_data_merc[outside_hull_2d] = np.nan

    # --------------------------
    # WebP speichern
    # --------------------------
    outname = f"{var_type}_{valid_time_local:%Y%m%d_%H%M}.webp" if valid_time_local else f"{var_type}_unknown.webp"
    out_path = os.path.join(output_dir, outname)
    save_transparent_webp(render_data_merc, cmap, norm, out_path)

    print(f"{filename} -> {outname}")

    # Aufräumen
    del data, render_data, render_data_merc
    gc.collect()

print("Fertig!")
