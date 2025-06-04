# IMPORTANT: This script works only when executing it from a device in the database's host vLAN.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import date


def query_silos_df():
    conn = psycopg2.connect(database="meteo",
                            host="172.26.201.28",
                            user="postgres",
                            password="0000",
                            port="5432")
    cursor = conn.cursor()

    cursor.execute("""
                   SELECT silo_id, silo_name, longitude, latitude
                   FROM silos;
                   """)
    answ = cursor.fetchall()

    cursor.close()
    conn.close()
    df_silos = pd.DataFrame(answ,
                            columns=["silo_id",
                                     "silo_name",
                                     "longitude",
                                     "latitude"])
    return df_silos

def query_weather_df(initial_date = "2019-01-01", final_date = "2024-12-31"):
    conn = psycopg2.connect(database="meteo",
                            host="172.26.201.28",
                            user="postgres",
                            password="0000",
                            port="5432")
    cursor = conn.cursor()

    cursor.execute(f"""
                   SELECT silo_id, date_mesure, T2M_MAX, T2M_MEAN, T2M_MIN, SSI_MEAN, PRECIP_SUM, WS2M_MEAN, DEWT2M_MEAN
                   FROM daily_weather
                   WHERE date_mesure >= '{initial_date}' and date_mesure <= '{final_date}'
                   ;
                   """)
    answ = cursor.fetchall()
    cursor.close()
    conn.close()
    df_weather = pd.DataFrame(answ, columns=["silo_id", "date_mesure", "T2M_MAX",
                                             "T2M_MEAN", "T2M_MIN", "SSI_MEAN",
                                             "PRECIP_SUM", "WS2M_MEAN", "DEWT2M_MEAN"
                                             ])
    return df_weather

def merge_export_df(silos_df, weather_df, output_file):
    df = pd.merge(left=silos_df,
                  right=weather_df,
                  how="left",
                  on="silo_id")
    df = df.reindex(["silo_name", "date_mesure", "silo_id",
                     "longitude", "latitude", "T2M_MAX",
                     "T2M_MEAN", "T2M_MIN", "SSI_MEAN",
                     "PRECIP_SUM", "WS2M_MEAN", "DEWT2M_MEAN"],
                     axis=1)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_file)


def by_year_query_wrapper(year):
    df_silos = query_silos_df()
    df_weather = query_weather_df(initial_date = f"{year-1}-01-01",
                                  final_date = f"{year}-12-31")
    merge_export_df(df_silos, df_weather,
                    "src/raw_data/multi_meteo/Agrial_meteo_{}_{}.parquet".format(year, date.today().strftime("%d.%m.%Y")))
# if __name__ == "__main__":
for year in range(2019, 2025):
    by_year_query_wrapper(year)
#     df_silos = query_silos_df()
#     df_weather = query_weather_df()
#     df = merge_export_df(df_silos, df_weather,
#                          "src/raw_data/Agrial_meteo_{}.parquet".format(date.today().strftime("%d.%m.%Y")))
