import sys
import cfgrib
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import os
from adjustText import adjust_text
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
from zoneinfo import ZoneInfo
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from scipy.interpolate import NearestNDInterpolator
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# ------------------------------
# Eingabe-/Ausgabe
# ------------------------------
data_dir = sys.argv[1]
output_dir = sys.argv[2]
var_type = sys.argv[3].lower()
gridfile = sys.argv[4] if len(sys.argv) > 4 else "data/grid/grid.nc"
cape_dir = sys.argv[5] if len(sys.argv) > 5 else None
wshearu_dir = sys.argv[6] if len(sys.argv) > 6 else None
wshearv_dir = sys.argv[7] if len(sys.argv) > 7 else None

if not os.path.exists(gridfile):
    raise FileNotFoundError(f"Grid-Datei nicht gefunden: {gridfile}")
    
os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# Geo-Daten
# ------------------------------
cities = pd.DataFrame({
    'name': ['Berlin', 'Hamburg', 'München', 'Köln', 'Frankfurt', 'Dresden', 'Stuttgart', 'Düsseldorf',
             'Nürnberg', 'Erfurt', 'Leipzig', 'Bremen', 'Saarbrücken', 'Hannover', 'Magdeburg'],
    'lat': [52.52, 53.55, 48.14, 50.94, 50.11, 51.05, 48.78, 51.23,
            49.45, 50.98, 51.34, 53.08, 49.24, 52.37, 52.13],
    'lon': [13.40, 9.99, 11.57, 6.96, 8.68, 13.73, 9.18, 6.78,
            11.08, 11.03, 12.37, 8.80, 6.99, 9.73, 11.62]
})


# ------------------------------
# Farben und Normen
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
ww_categories = {
    "Bewölkung": [0, 1, 2, 3],
    "Nebel": [48, 45],
    "Schneeregen": [56, 57],
    "Regen": [61, 63, 65],
    "gefr. Regen": [66, 67],
    "Schnee": [71, 73, 75],
    "Gewitter": [95, 96],
}

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

prec_bounds = [0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
               12, 14, 16, 20, 24, 30, 40, 50, 60, 80, 100, 125]
prec_colors = ListedColormap([
    "#B4D7FF", "#75BAFF", "#349AFF", "#0582FF", "#0069D2",
    "#003680", "#148F1B", "#1ACF06", "#64ED07", "#FFF32B",
    "#E9DC01", "#F06000", "#FF7F26", "#FFA66A", "#F94E78",
    "#F71E53", "#BE0000", "#880000", "#64007F", "#C201FC",
    "#DD66FE", "#EBA6FF", "#F9E7FF", "#D4D4D4", "#969696"
])
prec_norm = BoundaryNorm(prec_bounds, prec_colors.N)

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
dbz_norm = mcolors.BoundaryNorm(dbz_bounds, dbz_colors.N)

# ------------------------------
# Aufsummierter Niederschlag (tp_acc)
# ------------------------------
tp_acc_bounds = [0.1, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100,
                 125, 150, 175, 200, 250, 300, 400, 500]
tp_acc_colors = ListedColormap([
    "#B4D7FF", "#75BAFF", "#349AFF", "#0582FF", "#0069D2",
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

#-------------------------------
# Schneehöhen-Farben
#------------------------------
snow_bounds = [0, 0.1, 0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200, 250, 300, 400]
snow_colors = ListedColormap([
        "#F8F8F8", "#DCDBFA", "#AAA9C8", "#75BAFF", "#349AFF", "#0582FF",
        "#0069D2", "#004F9C", "#01327F", "#4B007F", "#64007F", "#9101BB",
        "#C300FC", "#D235FF", "#EBA6FF", "#F4CEFF", "#FAB2CA", "#FF9798",
        "#FE6E6E", "#DF093F", "#BE0000", "#A40000", "#880000", "#460000"
    ])
snow_norm = mcolors.BoundaryNorm(snow_bounds, snow_colors.N)

#-------------------------------
# SRH-Farben
#------------------------------
srh_bounds = [-250, -200, -150, -100, -50, -25, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 1000, 1250, 1500]
srh_colors = ListedColormap([
        "#0069D2", "#0482FF", "#359AFF", "#75BAFF", "#D2E9FF", "#FFFFFF",
        "#B4FF5A", "#63ED07", "#1ACF05", "#97C90E", "#E8DC00", "#FFF42B",
        "#FFA66A", "#F84E78", "#F71E54", "#BF0000", "#880000", "#64007F",
        "#C200FB", "#DD66FF", "#EBA6FF", "#EFC7FA"
    ])
srh_norm = mcolors.BoundaryNorm(srh_bounds, srh_colors.N)

#-------------------------------
# EHI-Farben
#------------------------------
ehi_bounds = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
ehi_colors = ListedColormap([
    "#FFFFFF", "#75BAFF", "#0482FF", "#1ACF05", "#63ED07",
    "#FFF42B", "#E8DC00", "#FF7F27", "#F71E54", "#880000",
    "#64007F", "#C200FB", "#DD66FF", "#EBA6FF", "#B97A57"
])
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
# Kartenparameter
# ------------------------------
FIG_W_PX, FIG_H_PX = 880, 830
BOTTOM_AREA_PX = 179
TOP_AREA_PX = FIG_H_PX - BOTTOM_AREA_PX
TARGET_ASPECT = FIG_W_PX / TOP_AREA_PX

extent = [5, 16, 47, 56]

# ------------------------------
# WW-Legende Funktion
# ------------------------------
def add_ww_legend_bottom(fig, ww_categories, ww_colors_base):
    legend_height = 0.12
    legend_ax = fig.add_axes([0.05, 0.01, 0.9, legend_height])
    legend_ax.axis("off")
    for i, (label, codes) in enumerate(ww_categories.items()):
        n_colors = len(codes)
        block_width = 1.0 / len(ww_categories)
        gap = 0.05 * block_width
        x0 = i * block_width
        x1 = (i + 1) * block_width
        inner_width = x1 - x0 - gap
        color_width = inner_width / n_colors
        for j, c in enumerate(codes):
            color = ww_colors_base.get(c, "#FFFFFF")
            legend_ax.add_patch(mpatches.Rectangle((x0 + j * color_width, 0.3),
                                                  color_width, 0.6,
                                                  facecolor=color, edgecolor='black'))
        legend_ax.text((x0 + x1)/2, 0.05, label, ha='center', va='bottom', fontsize=10)

# ------------------------------
# ICON Grid laden (einmal!)
# ------------------------------
nc = netCDF4.Dataset(gridfile)
lats = np.rad2deg(nc.variables["clat"][:])
lons = np.rad2deg(nc.variables["clon"][:])
nc.close()


# ------------------------------
# Dateien durchgehen
# ------------------------------
for filename in sorted(os.listdir(data_dir)):
    if not filename.endswith(".grib2"):
        continue
    path = os.path.join(data_dir, filename)
    ds = cfgrib.open_dataset(path)

    # --------------------------
    # Daten je Typ
    # --------------------------
    if var_type == "t2m":
        if "t2m" not in ds: continue
        data = ds["t2m"].values - 273.15
        cmap, norm = t2m_colors, t2m_norm
    elif var_type == "tp":
        if "tprate" not in ds: continue
        data = ds["tprate"].values
        data[data < 0.1] = 0
        cmap, norm = prec_colors, prec_norm
        cmap.set_under('none')
    elif var_type == "ww":
        varname = next((vn for vn in ds.data_vars if vn in ["WW", "weather"]), None)
        if varname is None:
            print(f"Keine WW in {filename}")
            continue
        data = ds[varname].values
        cmap = None
    elif var_type == "dbz_cmax":
        if "DBZ_CMAX" not in ds: continue
        data = ds["DBZ_CMAX"].values
        cmap, norm = dbz_colors, dbz_norm
    elif var_type == "wind":
        if "fg10" not in ds:
            print(f"Keine wind-Variable in {filename} ds.keys(): {list(ds.keys())}")
            continue
        data = ds["fg10"].values
        data = data * 3.6  # m/s → km/h
        cmap, norm = wind_colors, wind_norm
    elif var_type == "tp_acc":
        if "tp" not in ds: continue
        data = ds["tp"].values
        data[data < 0.1] = 0
        cmap, norm = tp_acc_colors, tp_acc_norm
        cmap.set_under('none')
    elif var_type == "cape_ml":
        if "CAPE_ML" not in ds: continue
        data = ds["CAPE_ML"].values
        data[data < 0] = np.nan
        cmap, norm = cape_colors, cape_norm
    elif var_type == "srh3km":
        if "hlcy" not in ds:
            print(f"Keine srh-Variable in {filename} ds.keys(): {list(ds.keys())}")
            continue
        data = ds["hlcy"].values
        cmap, norm = srh_colors, srh_norm
    elif var_type == "snow":
        if "sde" not in ds: continue
        data = ds["sde"].values * 100
        cmap, norm = snow_colors, snow_norm
    elif var_type == "ehi":
        if cape_dir is None:
            print("Kein cape_dir angegeben (5. Argument)")
            continue
        # srh3km_000.grib2 -> cape_000.grib2
        suffix = filename.replace("srh3km_", "")  # z.B. "000.grib2"
        cape_filename = f"cape_ml_{suffix}"
        cape_path = os.path.join(cape_dir, cape_filename)
        if not os.path.exists(cape_path):
            print(f"Kein passendes CAPE-File gefunden: {cape_path}")
            continue
        ds_cape = cfgrib.open_dataset(cape_path)
        if "hlcy" not in ds:
            print(f"hlcy fehlt in {filename}")
            continue
        if "CAPE_ML" not in ds_cape:
            print(f"CAPE_ML fehlt in {cape_filename}")
            continue
        srh = ds["hlcy"].values
        cape = ds_cape["CAPE_ML"].values
        cape[cape < 0] = 0
        data = (cape * srh) / 160000.0
        data[data < 0] = np.nan
        cmap, norm = ehi_colors, ehi_norm
    elif var_type == "scp":
        if None in [cape_dir, wshearu_dir, wshearv_dir]:
            print("Für SCP werden cape_dir, wshearu_dir, wshearv_dir benötigt")
            continue

        suffix = filename.replace("srh3km_", "")

        cape_path    = os.path.join(cape_dir,    f"cape_ml_{suffix}")
        wshearu_path = os.path.join(wshearu_dir, f"wshear_u_{suffix}")
        wshearv_path = os.path.join(wshearv_dir, f"wshear_v_{suffix}")

        missing = False
        for p in [cape_path, wshearu_path, wshearv_path]:
            if not os.path.exists(p):
                print(f"Datei nicht gefunden: {p}")
                missing = True
        if missing:
            continue

        ds_cape    = cfgrib.open_dataset(cape_path)
        ds_wshearu = cfgrib.open_dataset(wshearu_path)
        ds_wshearv = cfgrib.open_dataset(wshearv_path)

        if "hlcy" not in ds:
            print(f"hlcy fehlt in {filename}")
            continue

        srh  = ds["hlcy"].values                  # 0-3km SRH aus data_dir
        cape = ds_cape["CAPE_ML"].values
        bs06 = np.sqrt(ds_wshearu["WSHEAR_U"].values**2 + ds_wshearv["WSHEAR_V"].values**2)

        cape[cape < 0] = 0
        cape_term = cape / 1000
        srh_term  = np.maximum(0, srh) / 50
        ebs_term  = np.where(bs06 < 10, 0.0, np.where(bs06 > 20, 1.0, bs06 / 20))

        data = np.round(cape_term * srh_term * ebs_term, 2)
        data[data < 0] = np.nan
        cmap, norm = scp_colors, scp_norm
    else:
        print(f"Var_type {var_type} nicht implementiert")
        continue

    if data.ndim == 3: data = data[0]

    # --------------------------
    # Zeiten
    # --------------------------
    run_time_utc = pd.to_datetime(ds["time"].values) if "time" in ds else None
    if "valid_time" in ds:
        valid_time_raw = ds["valid_time"].values
        valid_time_utc = pd.to_datetime(valid_time_raw[0]) if np.ndim(valid_time_raw) > 0 else pd.to_datetime(valid_time_raw)
    else:
        step = pd.to_timedelta(ds["step"].values[0])
        valid_time_utc = run_time_utc + step
    valid_time_local = valid_time_utc.tz_localize("UTC").astimezone(ZoneInfo("Europe/Berlin"))

    # --------------------------
    # Figure
    # --------------------------
    scale = 0.9
    fig = plt.figure(figsize=(FIG_W_PX/100*scale, FIG_H_PX/100*scale), dpi=100)
    shift_up = 0.02
    ax = fig.add_axes([0.0, BOTTOM_AREA_PX/FIG_H_PX + shift_up, 1.0, TOP_AREA_PX/FIG_H_PX],
                    projection=ccrs.PlateCarree())
    ax.set_extent(extent)
    ax.set_axis_off()
    ax.set_aspect('auto')

    # ------------------------------
    # Regelmäßiges Gitter definieren
    # ------------------------------
    lon_min, lon_max, lat_min, lat_max = extent
    res = 0.015
    lon_grid = np.arange(lon_min, lon_max + res, res)
    lat_grid = np.arange(lat_min, lat_max + res, res)
    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)

    # ------------------------------
    # Interpolation auf regelmäßiges Gitter
    # ------------------------------
    points = np.column_stack((lons, lats))
    valid_mask = np.isfinite(data)
    points_valid = points[valid_mask]
    data_valid = data[valid_mask]

    interpolator = NearestNDInterpolator(points_valid, data_valid)
    data_grid = interpolator(lon_grid, lat_grid)

    # ------------------------------
    # pcolormesh Plot
    # ------------------------------
    if cmap is not None:
        im = ax.pcolormesh(lon_grid, lat_grid, data_grid, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        if var_type == "dbz_cmax":
            data_smooth = gaussian_filter(data_grid, sigma=0.8)
            im = ax.pcolormesh(lon_grid, lat_grid, data_smooth, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    else:
        # WW-Farben
        valid_mask = np.isfinite(data)
        codes = np.unique(data[valid_mask]).astype(int)
        codes = [c for c in codes if c in ww_colors_base]
        codes.sort()
        cmap = ListedColormap([ww_colors_base[c] for c in codes])
        code2idx = {c: i for i, c in enumerate(codes)}
        idx_data = np.full_like(data_grid, fill_value=np.nan, dtype=float)
        for c, i in code2idx.items():
            idx_data[data_grid == c] = i
        im = ax.pcolormesh(lon_grid, lat_grid, idx_data, cmap=cmap, vmin=-0.5, vmax=len(codes)-0.5, transform=ccrs.PlateCarree())

    ax.add_feature(cfeature.STATES.with_scale("10m"), edgecolor="#2C2C2C", linewidth=1)

    for _, city in cities.iterrows():
        ax.plot(city["lon"], city["lat"], "o", markersize=6, markerfacecolor="black",
                markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        txt = ax.text(city["lon"]+0.1, city["lat"]+0.1, city["name"],
                    fontsize=9, color="black", weight="bold", zorder=6)
        txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="white")])
    if var_type == "t2m":
        margin = 0.5
        data_masked = np.where(
            (lon_grid >= extent[0] + margin) & (lon_grid <= extent[1] - margin) &
            (lat_grid >= extent[2] + margin) & (lat_grid <= extent[3] - margin),
            data_grid, np.nan
        )

        max_idx = np.unravel_index(np.nanargmax(data_masked), data_masked.shape)

        for idx in [max_idx]:
            val = data_masked[idx]
            lon = lon_grid[idx]
            lat = lat_grid[idx]
            txt = ax.text(lon, lat, f"{val:.0f}",
                        fontsize=14, color="white", fontweight="bold",
                        ha="center", va="center", zorder=11,
                        clip_on=True,
                        transform=ccrs.PlateCarree())
            txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="black")])
    elif var_type == "wind":
        margin = 0.5
        data_masked = np.where(
            (lon_grid >= extent[0] + margin) & (lon_grid <= extent[1] - margin) &
            (lat_grid >= extent[2] + margin) & (lat_grid <= extent[3] - margin),
            data_grid, np.nan
        )

        max_idx = np.unravel_index(np.nanargmax(data_masked), data_masked.shape)

        for idx in [max_idx]:
            val = data_masked[idx]
            lon = lon_grid[idx]
            lat = lat_grid[idx]
            txt = ax.text(lon, lat, f"{val:.0f}",
                        fontsize=14, color="white", fontweight="bold",
                        ha="center", va="center", zorder=11,
                        clip_on=True,
                        transform=ccrs.PlateCarree())
            txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="black")])
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.COASTLINE)
    ax.add_patch(mpatches.Rectangle((0,0),1,1, transform=ax.transAxes, fill=False, color="black", linewidth=2))

    # --------------------------
    # Colorbar
    # --------------------------
    legend_h_px = 50
    legend_bottom_px = 45
    if var_type in ["t2m", "tp", "dbz_cmax", "tp_acc", "cape_ml", "snow", "srh3km", "ehi", "scp", "wind"]:
        bounds = (t2m_bounds if var_type == "t2m" else
                  prec_bounds if var_type == "tp" else
                  dbz_bounds if var_type == "dbz_cmax" else
                  tp_acc_bounds if var_type == "tp_acc" else
                  cape_bounds if var_type == "cape_ml" else
                  snow_bounds if var_type == "snow" else
                  ehi_bounds if var_type == "ehi" else
                  srh_bounds if var_type == "srh3km" else
                  scp_bounds if var_type == "scp" else
                  wind_bounds)
        cbar_ax = fig.add_axes([0.03, legend_bottom_px / FIG_H_PX, 0.94, legend_h_px / FIG_H_PX])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", ticks=bounds)
        cbar.ax.tick_params(colors="black", labelsize=7)
        cbar.outline.set_edgecolor("black")
        cbar.ax.set_facecolor("white")

        if var_type == "t2m":
            tick_labels = [str(tick) if tick % 4 == 0 else "" for tick in bounds]
            cbar.set_ticklabels(tick_labels)
        if var_type == "snow":
            cbar.set_ticklabels([int(tick) if float(tick).is_integer() else tick for tick in snow_bounds])
        if var_type == "tp":
            cbar.set_ticklabels([int(tick) if float(tick).is_integer() else tick for tick in prec_bounds])
        if var_type == "tp_acc":
            cbar.set_ticklabels([int(tick) if float(tick).is_integer() else tick for tick in tp_acc_bounds])
        if var_type == "scp":
            cbar.set_ticklabels([int(tick) if float(tick).is_integer() else tick for tick in scp_bounds])
        
    else:
        add_ww_legend_bottom(fig, ww_categories, ww_colors_base)

    # Footer
    footer_ax = fig.add_axes([0.0, (legend_bottom_px + legend_h_px)/FIG_H_PX, 1.0,
                            (BOTTOM_AREA_PX - legend_h_px - legend_bottom_px)/FIG_H_PX])
    footer_ax.axis("off")
    footer_texts = {
        "ww": "Signifikantes Wetter",
        "t2m": "Temperatur 2m (°C)",
        "tp": "Niederschlag, 1Std (mm)",
        "dbz_cmax": "Sim. Max. Radarreflektivität (dBZ)",
        "tp_acc": "Akkumulierter Niederschlag (mm)",
        "wind": "Windböen (km/h)",
        "cape_ml": "CAPE-Index (J/kg)",
        "snow": "Schneehöhe (cm)",
        "srh3km": "Helizität 0-3km (m²/s²)",
        "ehi": "Energy Helicity Index (EHI)",
        "scp": "Supercell Composite Parameter (SCP)",
    }

    left_text = footer_texts.get(var_type, var_type) + \
                f"\nICON-RUC ({pd.to_datetime(run_time_utc).hour:02d}z), Deutscher Wetterdienst" \
                if run_time_utc is not None else \
                footer_texts.get(var_type, var_type) + "\nICON-RUC (??z), Deutscher Wetterdienst"

    footer_ax.text(0.01, 0.85, left_text, fontsize=12, fontweight="bold", va="top", ha="left")
    footer_ax.text(0.734, 0.92, "Prognose für:", fontsize=12, va="top", ha="left", fontweight="bold")
    footer_ax.text(0.99, 0.68, f"{valid_time_local:%d.%m.%Y %H:%M} Uhr",
                fontsize=12, va="top", ha="right", fontweight="bold")

    # Speichern
    outname = f"{var_type}_{valid_time_local:%Y%m%d_%H%M}.png"
    plt.savefig(os.path.join(output_dir, outname), dpi=100, bbox_inches=None, pad_inches=0)
    plt.close()
