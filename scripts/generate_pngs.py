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

if not os.path.exists(gridfile):
    raise FileNotFoundError(f"Grid-Datei nicht gefunden: {gridfile}")
    
os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# Geo-Daten
# ------------------------------
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
ww_categories = {
    "Bewölkung": [0, 1, 2, 3],
    "Nebel": [45],
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

# ------------------------------
# Schneewahrscheinlichkeit 
# ------------------------------
snowprob_bounds = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
snowprob_colors = ListedColormap([
    "#FE9226", "#FFC02B", "#FFEE32", "#DDE02D", "#BBD629",
    "#9AC925", "#79BC21", "#37A319", "#367C40",
    "#366754", "#4A3E7C", "#593192"
])

snowprob_norm = mcolors.BoundaryNorm(snowprob_bounds, snowprob_colors.N)

#-------------------------------
# Schneehöhen-Farben
#------------------------------
snow_bounds = [0, 0.1, 0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200, 250, 300, 400]  # in cm
snow_colors = ListedColormap([
        "#F8F8F8", "#DCDBFA", "#AAA9C8", "#75BAFF", "#349AFF", "#0582FF",
        "#0069D2", "#004F9C", "#01327F", "#4B007F", "#64007F", "#9101BB",
        "#C300FC", "#D235FF", "#EBA6FF", "#F4CEFF", "#FAB2CA", "#FF9798",
        "#FE6E6E", "#DF093F", "#BE0000", "#A40000", "#880000", "#460000"
    ])
snow_norm = mcolors.BoundaryNorm(snow_bounds, snow_colors.N)

# ------------------------------
# Kartenparameter
# ------------------------------
FIG_W_PX, FIG_H_PX = 880, 830
BOTTOM_AREA_PX = 179
TOP_AREA_PX = FIG_H_PX - BOTTOM_AREA_PX
TARGET_ASPECT = FIG_W_PX / TOP_AREA_PX

# Bounding Box Deutschland (fix, keine GeoJSON nötig)
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
nc = netCDF4.Dataset(gridfile)  # Datei öffnen
lats = np.rad2deg(nc.variables["clat"][:])
lons = np.rad2deg(nc.variables["clon"][:])
nc.close()


# ------------------------------
# Dateien pro Step sammeln
# ------------------------------
import re
step_files = {}
for filename in os.listdir(data_dir):
    if filename.endswith(".grib2"):
        m = re.search(r"snow_(PT\d+H\d+M)_M(\d+)", filename)
        if m:
            step = m.group(1)
            member = int(m.group(2))
            step_files.setdefault(step, []).append((member, os.path.join(data_dir, filename)))

# Sortiere die Steps numerisch
def step_key(step_str):
    # PT000H00M → 0, PT014H00M → 14
    m = re.match(r"PT(\d+)H", step_str)
    return int(m.group(1)) if m else 0

# ------------------------------
# Schrittweise alle Member laden und auswerten (snow1cm/snow2cm)
# ------------------------------
if var_type in ["snow1cm", "snow2cm"]:
    for step, member_files in sorted(step_files.items(), key=lambda x: step_key(x[0])):
        # Member nach Nummer sortieren
        files = [f for _, f in sorted(member_files, key=lambda x: x[0])]

        
        threshold = 1 if var_type == "snow1cm" else 2

        # Erstes File nur für Zeitinformationen
        first_ds = cfgrib.open_dataset(files[0])
        run_time_utc = pd.to_datetime(first_ds["time"].values) if "time" in first_ds else None
        if "valid_time" in first_ds:
            valid_time_raw = first_ds["valid_time"].values
            valid_time_utc = pd.to_datetime(valid_time_raw[0]) if np.ndim(valid_time_raw) > 0 else pd.to_datetime(valid_time_raw)
        else:
            step_td = pd.to_timedelta(first_ds["step"].values[0])
            valid_time_utc = run_time_utc + step_td

        # Alle Members sammeln
        snow_list = []
        for path in files:
            ds = cfgrib.open_dataset(path)
            if "sde" not in ds:
                continue
            snow_m = ds["sde"].values * 100  # in cm umrechnen
            snow_list.append(snow_m)
            cmap, norm = snowprob_colors, snowprob_norm

        if len(snow_list) == 0:
            continue

        # In ein 2D-Array stapeln (Members x Grid-Punkte)
        snow_all = np.stack(snow_list, axis=0)

        # Wahrscheinlichkeit berechnen
        data = np.sum(snow_all >= threshold, axis=0) / snow_all.shape[0] * 100

        print(f"Step {step}: {snow_all.shape[0]} Ensemble-Member ausgewertet (≥{threshold}cm)")

        # Lokale Zeit
        valid_time_local = valid_time_utc.tz_localize("UTC").tz_convert(ZoneInfo("Europe/Berlin"))

        # ==============================
        # PLOT snow1cm / snow2cm
        # ==============================
        # ==============================
        # Figure
        # ==============================
        scale = 0.9
        fig = plt.figure(figsize=(FIG_W_PX/100*scale, FIG_H_PX/100*scale), dpi=100)
        shift_up = 0.02
        ax = fig.add_axes(
            [0.0, BOTTOM_AREA_PX/FIG_H_PX + shift_up, 1.0, TOP_AREA_PX/FIG_H_PX],
            projection=ccrs.PlateCarree()
        )
        ax.set_extent(extent)
        ax.set_axis_off()
        ax.set_aspect('auto')

        # ==============================
        # Regelmäßiges Gitter
        # ==============================
        lon_min, lon_max, lat_min, lat_max = extent
        res = 0.015
        lon_grid = np.arange(lon_min, lon_max + res, res)
        lat_grid = np.arange(lat_min, lat_max + res, res)
        lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)

        # ==============================
        # Interpolation (Nearest)
        # ==============================
        points = np.column_stack((lons, lats))
        valid_mask = np.isfinite(data)
        points_valid = points[valid_mask]
        data_valid = data[valid_mask]

        interpolator = NearestNDInterpolator(points_valid, data_valid)
        data_grid = interpolator(lon_grid, lat_grid)

        # ==============================
        # Plot
        # ==============================
        im = ax.pcolormesh(
            lon_grid, lat_grid, data_grid,
            cmap=snowprob_colors,
            norm=snowprob_norm,
            transform=ccrs.PlateCarree()
        )

        # Bundesländer, Grenzen
        ax.add_feature(cfeature.STATES.with_scale("10m"), edgecolor="#2C2C2C", linewidth=1)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.COASTLINE)

        # Städte
        for _, city in cities.iterrows():
            ax.plot(
                city["lon"], city["lat"], "o",
                markersize=6, markerfacecolor="black",
                markeredgecolor="white", markeredgewidth=1.5, zorder=5
            )
            txt = ax.text(
                city["lon"] + 0.1, city["lat"] + 0.1, city["name"],
                fontsize=9, color="black", weight="bold", zorder=6
            )
            txt.set_path_effects([
                path_effects.withStroke(linewidth=1.5, foreground="white")
            ])

        ax.add_patch(
            mpatches.Rectangle(
                (0, 0), 1, 1,
                transform=ax.transAxes,
                fill=False, color="black", linewidth=2
            )
        )

        # ==============================
        # Colorbar
        # ==============================
        legend_h_px = 50
        legend_bottom_px = 45

        cbar_ax = fig.add_axes([
            0.03,
            legend_bottom_px / FIG_H_PX,
            0.94,
            legend_h_px / FIG_H_PX
        ])

        cbar = fig.colorbar(
            im, cax=cbar_ax,
            orientation="horizontal",
            ticks=snowprob_bounds
        )

        cbar.ax.tick_params(labelsize=7)
        cbar.outline.set_edgecolor("black")
        cbar.ax.set_facecolor("white")

        # ==============================
        # Footer
        # ==============================
        footer_ax = fig.add_axes([
            0.0,
            (legend_bottom_px + legend_h_px) / FIG_H_PX,
            1.0,
            (BOTTOM_AREA_PX - legend_h_px - legend_bottom_px) / FIG_H_PX
        ])
        footer_ax.axis("off")

        footer_title = (
            "Schneehöhe ≥ 1 cm (%)"
            if var_type == "snow1cm"
            else "Schneehöhe ≥ 2 cm (%)"
        )

        left_text = (
            f"{footer_title}\n"
            f"ICON-RUC ({pd.to_datetime(run_time_utc).hour:02d}z), Deutscher Wetterdienst"
        )

        footer_ax.text(
            0.01, 0.85, left_text,
            fontsize=12, fontweight="bold",
            va="top", ha="left"
        )

        footer_ax.text(
            0.734, 0.92, "Prognose für:",
            fontsize=12, va="top",
            ha="left", fontweight="bold"
        )

        footer_ax.text(
            0.99, 0.68,
            f"{valid_time_local:%d.%m.%Y %H:%M} Uhr",
            fontsize=12, va="top",
            ha="right", fontweight="bold"
        )

        # ==============================
        # Speichern
        # ==============================
        outname = f"{var_type}_{valid_time_local:%Y%m%d_%H%M}.png"
        plt.savefig(
            os.path.join(output_dir, outname),
            dpi=100, bbox_inches=None, pad_inches=0
        )
        plt.close()

    
# ------------------------------
# Dateien durchgehen
# ------------------------------
else:
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
            data[data < 0.1] = 0  # Fix: Setze <0.1 zu 0 statt NaN, um Interpolation mit 0 zu ermöglichen
            cmap, norm = prec_colors, prec_norm
            cmap.set_under('none')  # Fix: Transparenz für Werte unter der untersten Bound (für trockene Gebiete)
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
        elif var_type == "tp_acc":
            if "tp" not in ds: continue
            data = ds["tp"].values
            data[data < 0.1] = 0  # Fix: Setze <0.1 zu 0 statt NaN, um Interpolation mit 0 zu ermöglichen
            cmap, norm = tp_acc_colors, tp_acc_norm
            cmap.set_under('none')  # Fix: Transparenz für Werte unter der untersten Bound (für trockene Gebiete)
        elif var_type == "cape_ml":
            if "CAPE_ML" not in ds: continue
            data = ds["CAPE_ML"].values
            data[data < 0] = np.nan
            cmap, norm = cape_colors, cape_norm
        elif var_type == "snow":
            if "sde" not in ds: continue
            data = ds["sde"].values * 100  # in cm umrechnen
            cmap, norm = snow_colors, snow_norm
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
        res = 0.015  # Auflösung in Grad (anpassbar, z. B. 0.05 für höhere Auflösung)
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

        # Nearest Neighbor Interpolation (schnell und ausreichend für viele Fälle)
        interpolator = NearestNDInterpolator(points_valid, data_valid)
        data_grid = interpolator(lon_grid, lat_grid)

        # ------------------------------
        # pcolormesh Plot
        # ------------------------------
        if cmap is not None:
            # Für Variablen mit vorgegebener Farbkarte (t2m, tp, dbz_cmax, tp_acc, cape_ml)
            im = ax.pcolormesh(lon_grid, lat_grid, data_grid, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
            if var_type == "dbz_cmax":
                data_smooth = gaussian_filter (data_grid, sigma = 0.8)
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

        # Bundesländer-Grenzen aus Cartopy (statt GeoJSON)
        ax.add_feature(cfeature.STATES.with_scale("10m"), edgecolor="#2C2C2C", linewidth=1)

        for _, city in cities.iterrows():
            ax.plot(city["lon"], city["lat"], "o", markersize=6, markerfacecolor="black",
                    markeredgecolor="white", markeredgewidth=1.5, zorder=5)
            txt = ax.text(city["lon"]+0.1, city["lat"]+0.1, city["name"],
                        fontsize=9, color="black", weight="bold", zorder=6)
            txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="white")])
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.COASTLINE)
        ax.add_patch(mpatches.Rectangle((0,0),1,1, transform=ax.transAxes, fill=False, color="black", linewidth=2))

        # --------------------------
        # Colorbar (falls relevant)
        # --------------------------
        legend_h_px = 50
        legend_bottom_px = 45
        if var_type in ["t2m", "tp", "dbz_cmax", "tp_acc", "cape_ml", "snow1cm", "snow2cm", "snow"]:
            bounds = t2m_bounds if var_type == "t2m" else prec_bounds if var_type == "tp" else dbz_bounds if var_type == "dbz_cmax" else tp_acc_bounds if var_type == "tp_acc" else cape_bounds if var_type == "cape_ml" else snowprob_bounds if var_type == "snow1cm" else snowprob_bounds if var_type == "snow2cm" else snow_bounds
            cbar_ax = fig.add_axes([0.03, legend_bottom_px / FIG_H_PX, 0.94, legend_h_px / FIG_H_PX])
            cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", ticks=bounds)
            cbar.ax.tick_params(colors="black", labelsize=7)
            cbar.outline.set_edgecolor("black")
            cbar.ax.set_facecolor("white")

            if var_type == "t2m":
                tick_labels = [str(tick) if tick % 4 == 0 else "" for tick in bounds]
                cbar.set_ticklabels(tick_labels)
            if var_type=="snow":
                cbar.set_ticklabels([int(tick) if float(tick).is_integer() else tick for tick in snow_bounds])

            if var_type == "tp":
                cbar.set_ticklabels([int(tick) if float(tick).is_integer() else tick for tick in prec_bounds])
            if var_type=="tp_acc":
                cbar.set_ticklabels([int(tick) if float(tick).is_integer() else tick for tick in tp_acc_bounds])
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
            "cape_ml": "CAPE-Index (J/kg)",
            "snow1cm": "Schneehöhe ≥ 1 cm (%)",
            "snow2cm": "Schneehöhe ≥ 2 cm (%)",
            "snow": "Schneehöhe (cm)"
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
