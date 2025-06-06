import pcse
import pickle
import random # No permanent import, i have to find a deterministic soil assignment
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


GEOFOLIA_WOF_MAP = {
    "Orge d'hiver": ("barley", "Spring_barley_301"),
    "Blé tendre d'hiver": ("wheat", "Winter_wheat_105"),
    "Maïs fourrage": ("maize", "Fodder_maize_nl"),
    "Colza oléagineux d'hiver": ("rapeseed", "Oilseed_rape_1001"),
    'Maïs grain': ("maize", "Grain_maize_202"),
    "Seigle d'hiver": ("wheat", "Winter_wheat_105"),
    "Pois protéagineux d'hiver": ("fababean", "Faba_bean_801"),
    "Triticale d'hiver": ("wheat", "Winter_wheat_105"),
    "Avoine blanche d'hiver": ("barley", "Spring_barley_301"),
    "Orge de printemps": ("barley", "Spring_barley_301"),
    "Tournesol": ("sunflower", "Sunflower_1101"),
    "Blé dur d'hiver": ("wheat", "Winter_wheat_105"), 
    "Blé améliorant d'hiver": ("wheat", "Winter_wheat_105"),
    "Avoine noire d'hiver": ("barley", "Spring_barley_301"), 
    "Féverole d'hiver": ("fababean", "Faba_bean_801"),
    "Pois protéagineux de ptps": ("fababean", "Faba_bean_801"),
    "Sarrasin": ("wheat", "Winter_wheat_105"),
    "Millet": ("millet", "Millet_VanHeemst_1988"),
    "Sorgho grain": ("sorghum", "Sorghum_VanHeemst_1988"),
    "Orge 6 rangs d'hiver": ("barley", "Spring_barley_301"),
    "Maïs Epi": ("maize", "Grain_maize_202"),
    "Féverole de printemps": ("fababean", "Faba_bean_801"),
    "Orge 2 rangs de printemps": ("barley", "Spring_barley_301"),
    "Pois fourrager d'hiver": ("fababean", "Faba_bean_801"),
    "Sorgho fourrage": ("sorghum", "Sorghum_VanHeemst_1988"),
    "Blé tendre de printemps": ("wheat", "Winter_wheat_105"),
    "Avoine noire de printemps": ("barley", "Spring_barley_301"),
    "Colza fourrager printemps": ("rapeseed", "Oilseed_rape_1001"),
    "Seigle hiver": ("wheat", "Winter_wheat_105"),
}

def generate_simulations_df(plots_df_path):
    table = pq.read_table(plots_df_path) 
    base_df = table.to_pandas()
    sims = pd.DataFrame({"id": base_df["PlotId"],
                         "crop": base_df["CropName"].apply(lambda x: GEOFOLIA_WOF_MAP[x][0]),
                         "variety": base_df["CropName"].apply(lambda x: GEOFOLIA_WOF_MAP[x][1]),
                         "soil": [random.choice(["ec1", "ec2", "ec3", "ec4", "ec5", "ec6"]) for _ in range(len(base_df))], # This gonna change
                        #  "site": ,
                         "crop_start_date": pd.to_datetime(base_df["SowingDate"], format = "%d/%m/%Y %H:%M:%S", errors='coerce'),
                         "crop_end_date": pd.to_datetime(base_df["HarvestingDate"], format = "%d/%m/%Y %H:%M:%S", errors='coerce'),
                         })
    sims["site"] = "wofost_data/sites_data/mean_site.YAML"
    sims["weather"] = f"wofost_data/meteo_data/{sims['id']}.csv"
    sims["real_crop"] = base_df["CropName"]
    sims["RealizedYield"] = base_df["RealizedYield"]
    # TODO: Remove this to work with the entire data
    print("Using a subset of the dataset for testing purposes.")
    print(f"Total simulations to run: {len(sims)}")
    print(f"Actual simulations to run: {len(sims)//4}")
    index_list = random.choices(range(len(sims)), k=len(sims)//4)
    sims = sims.iloc[index_list, :]
    # sims = sims.iloc[:100, :]  # For testing purposes, only 100 simulations
    ## END TODO
    with open("src/sims_setup.pickle", "wb") as f:
        pickle.dump(obj=sims, file=f)

if __name__ == "__main__":
    plots_df_path = "src/raw_data/COORDS_pro_parcelles_02.06.2025.parquet" #TODO: Attention this path is dynamic.
    generate_simulations_df(plots_df_path)