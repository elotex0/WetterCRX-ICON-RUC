import sys
import cfgrib
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import pandas as pd
import os
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
from zoneinfo import ZoneInfo
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# ------------------------------
# Eingabe-/Ausgabe
# ------------------------------
data_dir = sys.argv[1]
output_dir = sys.argv[2]
var_type = sys.argv[3].lower()
os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# Geo-Daten
# ------------------------------
bundeslaender = gpd.read_file("scripts/bundeslaender.geojson")
cities = pd.DataFrame({
    'name': ['Berlin', 'Hamburg', 'München', 'Köln', 'Frankfurt', 'Dresden',
             'Stuttgart', 'Düsseldorf', 'Nürnberg', 'Erfurt', 'Leipzig',
             'Bremen', 'Saarbrücken', 'Hannover'],
    'lat': [52.52, 53.55, 48.14, 50.94, 50.11, 51.05, 48.78, 51.23,
            49.45, 50.98, 51.34, 53.08, 49.24, 52.37],
    'lon': [13.40, 9.99, 11.57, 6.96, 8.68, 13.73, 9.18, 6.78,
            11.08, 11.03, 12.37, 8.80, 6.99, 9.73]
})

# ------------------------------
# Farben und Normen
# ------------------------------

ignore_codes = {4}

# ------------------------------
# WW-Farben
# ------------------------------
ww_colors_base = {
    0: "#FFFFFF", 1: "#D3D3D3", 2: "#A9A9A9", 3: "#696969",
    45: "#FFFF00", 48: "#FFD700",
    56: "#FFA500", 57: "#C06A00",
    51: "#A3FFA3", 53: "#33FF33", 55: "#006600",
    61: "#33FF33", 63: "#009900", 65: "#006600",
    80: "#33FF33", 81: "#009900", 82: "#006600",
    66: "#FF6347", 67: "#8B0000",
    71: "#ADD8E6", 73: "#6495ED", 75: "#00008B",
    85: "#ADD8E6", 86: "#6495ED",
    77: "#ADD8E6",
    95: "#FF77FF", 96: "#C71585", 99: "#C71585"
}
ww_categories = {
    "Bewölkung": [0, 1 , 2, 3],
    "Nebel": [45],
    "Schneeregen": [56, 57],
    "Regen": [51, 61, 63, 65],
    "gefr. Regen": [66, 67],
    "Schnee": [71, 73, 75],
    "Gewitter": [95,96],
}

# Temperatur 2m
t2m_bounds = list(range(-28, 41, 2))
t2m_colors = [
    "#C802CB", "#AA00A9", "#8800AA", "#6600AA", "#4400AB",
    "#2201AA", "#0000CC", "#0033CC", "#0044CB", "#0055CC",
    "#0066CB", "#0076CD", "#0088CC", "#0099CB", "#00A5CB",
    "#00BB22", "#11C501", "#32D500", "#77D600", "#87DD00",
    "#FFCC00", "#FFBB00", "#FFAA01", "#FE9900", "#FF8800",
    "#FF6600", "#FF3300", "#FE0000", "#DC0000", "#BA0100",
    "#91002B", "#980065", "#BB0099", "#EE01AB", "#FF21FE"
]
t2m_cmap = ListedColormap(t2m_colors)
t2m_norm = BoundaryNorm(t2m_bounds, t2m_cmap.N)

# Niederschlag 1h
prec_bounds = [0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
               12, 14, 16, 20, 24, 30, 40, 50, 60, 80, 100, 125]
prec_colors = ListedColormap([
    "#B4D7FF","#75BAFF","#349AFF","#0582FF","#0069D2",
    "#003680","#148F1B","#1ACF06","#64ED07","#FFF32B",
    "#E9DC01","#F06000","#FF7F26","#FFA66A","#F94E78",
    "#F71E53","#BE0000","#880000","#64007F","#C201FC",
    "#DD66FE","#EBA6FF","#F9E7FF","#D4D4D4","#969696"
])
prec_norm = BoundaryNorm(prec_bounds, prec_colors.N)

# ------------------------------
# Kartenparameter
# ------------------------------
FIG_W_PX, FIG_H_PX = 880, 830
BOTTOM_AREA_PX = 179
TOP_AREA_PX = FIG_H_PX - BOTTOM_AREA_PX
TARGET_ASPECT = FIG_W_PX / TOP_AREA_PX
_minx, _miny, _maxx, _maxy = bundeslaender.total_bounds
_w, _h = _maxx - _minx, _maxy - _miny
ymin, ymax = _miny, _maxy
left_pad_factor, right_pad_factor = 0.56, 0.34
xmin = _minx - _w * left_pad_factor
xmax = _maxx + _w * right_pad_factor
needed_w = _h * TARGET_ASPECT
current_w = xmax - xmin
if current_w < needed_w:
    extra = (needed_w - current_w) / 2
    xmin -= extra
    xmax += extra
extent = [xmin, xmax, ymin, ymax]


#------------------------------
# WW Legende Funktion
#------------------------------

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
# Grid laden (ICON unstrukturiert)
# ------------------------------
gridfile = "scripts/icon_grid_0047_R19B07_L.nc"
grid = xr.open_dataset(gridfile)
lats = np.rad2deg(grid["clat"].values)
lons = np.rad2deg(grid["clon"].values)

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
        cmap, norm = t2m_cmap, t2m_norm
    elif var_type == "tp":
        if "tprate" not in ds: continue
        data = ds["tprate"].values
        data[data<0.1]=np.nan
        cmap, norm = prec_colors, prec_norm
    else:
        print(f"Var_type {var_type} nicht implementiert")
        continue

    if data.ndim==3: data=data[0]

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
    ax.add_feature(cfeature.LAND, facecolor="#676767")
    ax.add_feature(cfeature.OCEAN, facecolor="#676767")

    # Scatter Plot
    im = ax.scatter(lons, lats, c=data, s=2, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

    # Bundesländer & Städte
    bundeslaender.boundary.plot(ax=ax, edgecolor="black", linewidth=1)
    for _, city in cities.iterrows():
        ax.plot(city["lon"], city["lat"], "o", markersize=6, markerfacecolor="black",
                markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        txt = ax.text(city["lon"]+0.1, city["lat"]+0.1, city["name"],
                      fontsize=9, color="black", weight="bold", zorder=6)
        txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="white")])
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.COASTLINE)
    ax.add_patch(mpatches.Rectangle((0,0),1,1, transform=ax.transAxes, fill=False, color="black", linewidth=2))

    # Legende
    legend_h_px, legend_bottom_px = 50, 45
    cbar_ax = fig.add_axes([0.03, legend_bottom_px/FIG_H_PX, 0.94, legend_h_px/FIG_H_PX])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", extend='neither')
    cbar.ax.tick_params(colors="black", labelsize=7)
    cbar.outline.set_edgecolor("black")
    cbar.ax.set_facecolor("white")

    # Footer
    footer_ax = fig.add_axes([0.0, (legend_bottom_px + legend_h_px)/FIG_H_PX, 1.0,
                              (BOTTOM_AREA_PX - legend_h_px - legend_bottom_px)/FIG_H_PX])
    footer_ax.axis("off")
    footer_texts = {
        "ww": "Signifikantes Wetter",
        "t2m": "Temperatur 2m (°C)",
        "tp": "Niederschlag, 1Std (mm)",
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
