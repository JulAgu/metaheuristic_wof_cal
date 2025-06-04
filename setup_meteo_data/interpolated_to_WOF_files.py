import os
import math
import pandas as pd
import pyarrow.parquet as pq
import random
import csv
from joblib import Parallel, delayed, effective_n_jobs
from joblib_progress import joblib_progress

Wh_to_kJ = lambda x: x * 86.4

def ea_from_tdew(tdew):
    if tdew < -95.0 or tdew > 65.0:
        raise ValueError(f'tdew={tdew} is not in range -95 to +60 deg C')
    tmp = (17.27 * tdew) / (tdew + 237.3)
    return 0.6108 * math.exp(tmp)

tdew_to_kpa = lambda x: ea_from_tdew(x)

def plots_coords_to_WOFcsv(df_plot):
    df_pcse = pd.DataFrame({
        "DAY": df_plot["date_mesure"].dt.strftime("%Y%m%d"),  # YYYYMMDD
        "TMAX": df_plot["T2M_MAX"],  # °C
        "TMIN": df_plot["T2M_MIN"],  # °C
        "TEMP": df_plot["T2M_MEAN"],  # °C
        "IRRAD": df_plot["SSI_MEAN"].apply(Wh_to_kJ).round(2),  # kJ/m^2
        "RAIN": df_plot["PRECIP_SUM"],  # mm
        "WIND": df_plot["WS2M_MEAN"],
        "VAP": df_plot["DEWT2M_MEAN"].apply(tdew_to_kpa).round(2),  # kPa
        "SNOWDEPTH": "NaN",  # cm
    })

    lon = df_plot["Longitude"].unique()
    lat = df_plot["Latitude"].unique()
    plot_id = df_plot["PlotId"].unique()

    if len(lon) != 1 or len(lat) != 1 or len(plot_id) != 1:
        raise ValueError("Inconsistent coordinates or plot IDs in group")

    plot_info = {"ID": plot_id[0], "LON": lon[0], "LAT": lat[0], "ELEV": 30} # It shoul be awesome to have a real elevation here
    return df_pcse, plot_info

def fill_csv(year, plot_id, plot_info, df_psce, base_template):
    rows = [[row[0].format(plot_id)] if '{}' in row[0] else [row[0]] for row in base_template]
    rows.append([f"Longitude = {plot_info['LON']}; Latitude = {plot_info['LAT']}; Elevation = {plot_info['ELEV']}; AngstromA = 0.18; AngstromB = 0.55; HasSunshine = False"])
    rows.append(["## Daily weather observations (missing values are NaN)"])
    rows.append(list(df_psce.columns))
    rows.extend(df_psce.astype(str).values.tolist())

    out_path = f"wofost_data/meteo_data/{plot_id}.csv"
    with open(out_path, "w", newline='') as f:
        csv.writer(f).writerows(rows)

def process_and_write(plot_id, df_plot, year, base_template):
    df_psce, plot_info = plots_coords_to_WOFcsv(df_plot)
    fill_csv(year, plot_id, plot_info, df_psce, base_template)

if __name__ == "__main__":
    print("Using all CPU cores:", effective_n_jobs(-1))

    for year in range(2020, 2025):  # TODO: Adjust years here
        out_dir = f"wofost_data/meteo_data"
        os.makedirs(out_dir, exist_ok=True)

        # Load full data
        table = pq.read_table(f"src/raw_data/PLOTS_WITH_COORDS_{year}_02.06.2025.parquet")
        df = table.to_pandas()

        # Load base template only once
        with open("src/templates/meteo_wofost.csv", "r", newline='') as f:
            base_template = list(csv.reader(f))

        grouped = list(df.groupby("PlotId"))
        # grouped = random.sample(grouped, 100)

        with joblib_progress(description="Parallel processing...", total=len(grouped)):
            Parallel(n_jobs=-1)(
                delayed(process_and_write)(plot_id, group.reset_index(), year, base_template)
                for plot_id, group in grouped
            )
