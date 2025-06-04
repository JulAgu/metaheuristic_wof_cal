import numpy as np
import pandas as pd
from datetime import date
from scipy.spatial import cKDTree
import pyarrow as pa
import pyarrow.parquet as pq
from joblib import Parallel, delayed, effective_n_jobs
from joblib_progress import joblib_progress


def meteo_to_df(input_path):
    """
    Returns
    -------
    work_df : pd.DataFrame
        The weather TS in a apndas dataframe
    """
    df = pd.read_parquet(input_path)
    work_df = df.loc[:, ["silo_id","date_mesure", "longitude",
                         "latitude", "T2M_MAX", "T2M_MEAN",
                         "T2M_MIN", "SSI_MEAN", "PRECIP_SUM",
                         "WS2M_MEAN", "DEWT2M_MEAN"]]
    return work_df
    
def plots_to_df(input_path):
    df = pd.read_parquet(input_path)
    return df


def idw_interpolation(obs, xy_targets, power=2):
    """
    Perform IDW (Inverse Distance Weighting) interpolation
    Parameters
    ----------
    obs: np.array
        NxM array of observation coordinates (id, lon, lat, f1, ..., fm)
    xy_targets: np.array
        Mx2 array of target coordinates (lon, lat)
    power: int
        power parameter for IDW (default: 2)
    Returns
    -------
    interpolated_array: np.array
        interpolated values at target points
    """
    tree = cKDTree(obs[:, 1:3])
    dists, idxs = tree.query(xy_targets, k=3)  # Use 5 nearest neighbors
    weights = 1 / (dists ** power + 1e-12)
    weights /= weights.sum(axis=1)[:, None]
    interpolated_array = np.zeros((len(xy_targets), obs.shape[1]-1)) # -1 Because in the inerpolated array we don't use the IDs
    interpolated_array[:, :2] = xy_targets[:, :2]
    for col_indx in range(obs[:, 3:].shape[1]):
        value = obs[:, 3:][:, col_indx]
        interpolated = np.round(np.sum(weights * value[idxs], axis=1), decimals=2)
        interpolated_array[:, col_indx+2] = interpolated
    return interpolated_array

def one_day_interpolation(date, weather_df, plots_df):
    daily_df = weather_df[weather_df['date_mesure'] == date]
    obs_array = daily_df.loc[:, ["silo_id", "longitude", "latitude",
                                 "T2M_MAX", "T2M_MEAN", "T2M_MIN",
                                 "SSI_MEAN", "PRECIP_SUM", "WS2M_MEAN", "DEWT2M_MEAN"]].to_numpy()
    objective_array = plots_df.loc[:, ["Longitude", "Latitude"]].to_numpy()
    interpolated_array = idw_interpolation(obs_array, objective_array)
    interpolated_df = pd.DataFrame(interpolated_array,
                                   columns=["Longitude", "Latitude", "T2M_MAX",
                                            "T2M_MEAN", "T2M_MIN", "SSI_MEAN",
                                            "PRECIP_SUM", "WS2M_MEAN", "DEWT2M_MEAN"
                                            ])
    interpolated_df["date_mesure"] = date
    interpolated_df["PlotId"] = plots_df["PlotId"].values
    interpolated_df["YearId"] = plots_df["YearId"].values
    return interpolated_df


def wrapper_year_interpolation(year, weather_path, plots_path, output_path, n_jobs=-1):
    """
    Due to RAM limitations I'll try to manage each interpolation on an annual basis. Following this procces:
        1. Creating Multiple parquet files: one for each campaing year t : [t-1, t]
        2. At each iteration of the current function I filter the plots_coords to retain only the plots on the campaing year
        3. I only interpolate over this filter plots and using the cropped weather data.
    """
    weather_df = meteo_to_df(weather_path)
    print("wheater DF shape:", weather_df.shape)
    plots_coords = plots_to_df(plots_path)
    plots_coords = plots_coords.loc[plots_coords["YearId"] == year, :]
    print("plots DF shape:", plots_coords.shape)

    unique_dates = weather_df["date_mesure"].unique()

    # Parallel processing over dates
    with joblib_progress("Parallel process track..."):
        all_interpolated = Parallel(n_jobs=n_jobs)(
            delayed(one_day_interpolation)(date, weather_df, plots_coords)
            for date in unique_dates
        )

    final_df = pd.concat(all_interpolated, ignore_index=True)
    final_df.to_parquet(output_path, index=False)
    print(f"Parallel interpolation complete. Saved to {output_path}")


if __name__ == "__main__":
    # An exemple of the daily parallel interpolation
    print("Using all cpu cores: ", effective_n_jobs(-1))
    for year in range(2020, 2025):
        wrapper_year_interpolation(year = year,
                                   weather_path=f"src/raw_data/multi_meteo/Agrial_meteo_{year}_13.05.2025.parquet",
                                   plots_path="src/raw_data/COORDS_pro_parcelles_02.06.2025.parquet",                                                               
                                   output_path = "src/raw_data/PLOTS_WITH_COORDS_{}_{}.parquet".format(year, date.today().strftime("%d.%m.%Y")),
                                   n_jobs=-1)