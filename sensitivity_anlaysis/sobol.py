import os
import numpy as np
import pandas as pd
import pcse
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from wof_tools.wofost_exec import disable_logging
from pcse.input import YAMLCropDataProvider
from pcse.input import CABOFileReader
from pcse.input import WOFOST73SiteDataProvider
from pcse.input import YAMLAgroManagementReader
from pcse.input import CSVWeatherDataProvider
from pcse.base import ParameterProvider
from pcse.models import Wofost73_WLP_CWB, Wofost73_PP
from SALib.sample import saltelli
from joblib import Parallel, delayed, effective_n_jobs
from joblib_progress import joblib_progress
from SALib.analyze import sobol
import pickle
import yaml

def set_up_full_problem(calc_second_order=False,
                        n_samples= 50):
    """
    Creates the problem dictionaries for sensitivity analysis.
    """
    RANGES_VAR = {"TSUM1": (100, 2000),
                  "TSUM2": (100, 2000),
                  "SPAN":  (10, 70),
                  "CFET": (0.1, 1.0),
                  "CVL": (0.1, 1.0),
                  "CVO": (0.1, 1.0),
                  "CVR": (0.1, 1.0),
                  "CVS": (0.1, 1.0),
                  "TBASE": (0, 15),
                  "TBASEM": (0, 15),
                  "VERNBASE": (5, 15),
                  "RDI": (9, 11),
                  "RDMCR": (60, 300),
                  "RGRLAI": (0.001, 0.8),
                  }

    RANGES_SOIL = {"K0": (10, 100),
                   "SOPE": (0.2, 15),
                   "KSUB": (0.1, 30),
                   "RDMSOL": (90, 150),
    #                "SMW": (0.01, 0.5),
    #                "SMFCF": (0.01, 0.6),
    #                "CRAIRC": (0.01, 0.1),
                   }
    
    problem = {
        "num_vars": len(RANGES_VAR) + len(RANGES_SOIL),
        "names": list(RANGES_VAR.keys()) + list(RANGES_SOIL.keys()),
        "bounds": [list(bounds) for bounds in RANGES_VAR.values()] + [list(bounds) for bounds in RANGES_SOIL.values()]
        }

    paramsets = saltelli.sample(problem, n_samples, calc_second_order=calc_second_order)
    print(f"Generated {len(paramsets)} parameter sets for sensitivity analysis.")

    return problem, paramsets

def one_simulation_sensitivity(params_row,
                               problem,
                               paramsets,
                               wofost_data_path="wofost_data/",
                               output_path="output"):
    target_results = []
    os.makedirs(output_path, exist_ok=True)
    with open("wof_tools/wofost_exec_templates.yaml") as f:
        templates = yaml.safe_load(f)
    
    agro_str = templates["agromanage"].format(
                 date_campaign=f"{str(params_row['crop_end_date'].year-1)}-01-01",
                 crop=params_row["crop"],
                 variety=params_row["variety"],
                 crop_start= params_row["crop_start_date"].strftime("%Y-%m-%d"),
                 crop_end=params_row["crop_end_date"].strftime("%Y-%m-%d")
             )

    agromanag_params = yaml.safe_load(agro_str)

    disable_logging()
    crop_params = YAMLCropDataProvider(fpath=f"{wofost_data_path}crops_data")
    crop_params.set_active_crop(params_row["crop"], params_row["variety"])
    soil_params = CABOFileReader(f"{wofost_data_path}soils_data/{params_row['soil']}.soil")
    site_params = WOFOST73SiteDataProvider(WAV=100, CO2=410.0)
    weatherdata = CSVWeatherDataProvider(f"{wofost_data_path}meteo_data/{params_row['id']}.csv")

    parameters = ParameterProvider(cropdata=crop_params,
                                    soildata=soil_params,
                                    sitedata=site_params
                                    )
    for i, paramset in (enumerate(paramsets)):
        parameters.clear_override()
        for name, value in zip(problem["names"], paramset):
            parameters.set_override(name, value)
        try:
            wofsim = Wofost73_WLP_CWB(parameters,
                                      weatherdata,
                                      agromanag_params)

            wofsim.run_till_terminate()
            output = wofsim.get_output()
            summary_output = wofsim.get_summary_output()
            target_result = summary_output[0]["TWSO"]
            if target_result is None:
                print("Target variable is not available in summary output!")
            target_results.append(target_result)
        except Exception as e:
            print(f"Simulation failed for {params_row['id']}, paramset {i}: {e}")
            target_results.append(np.nan)  # Append NaN if simulation fails
    return np.array(target_results)


def wrapper_row(row):
    problem, paramsets = set_up_full_problem(calc_second_order=True)
    results = one_simulation_sensitivity(
        row,
        problem,
        paramsets,
        wofost_data_path="wofost_data/",
        output_path="output"
    )
    return problem, results


def mean_sobol_indices(results):
    def stack(key):
        return np.mean(np.array([res[key] for res in results]), axis=0)

    def stack_matrix(key):
        arr = np.array([
            [[np.nan if v is None else v for v in row] for row in res[key]]
            for res in results
        ])
        return np.nanmean(arr, axis=0)

    return {
        'S1': stack('S1'),
        'S1_conf': stack('S1_conf'),
        'ST': stack('ST'),
        'ST_conf': stack('ST_conf'),
        'S2': stack_matrix('S2') if 'S2' in results[0] else None,
        'S2_conf': stack_matrix('S2_conf') if 'S2_conf' in results[0] else None
    }


if __name__ == "__main__":

    with open("src/sims_setup.pickle", 'rb') as f:
        sims_data = pickle.load(f)
    simulations = sims_data.to_dict(orient="records")

    with joblib_progress(description ="Parallel process track...",
                             total=len(simulations)):
        outputs = Parallel(n_jobs=70)(
            delayed(wrapper_row)(row) for row in simulations)

    all_problems, all_results = zip(*outputs)

    sensitivity_results = [sobol.analyze(problem, results, calc_second_order=True) for problem, results in zip(all_problems, all_results)]
    Si = mean_sobol_indices(sensitivity_results)
    problem = all_problems[0]
    with open("sensitivity_results.pkl", "wb") as f:
        pickle.dump({"problem": problem, "mean_results": Si}, f)