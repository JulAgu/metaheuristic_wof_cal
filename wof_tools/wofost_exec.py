import os
import yaml
import pickle
import pandas as pd
import logging
from pcse.input import YAMLCropDataProvider
from pcse.input import CABOFileReader
from pcse.input import WOFOST73SiteDataProvider
from pcse.input import CSVWeatherDataProvider
from pcse.base import ParameterProvider
from pcse.models import Wofost73_WLP_CWB, Wofost73_PP
from joblib import Parallel, delayed, effective_n_jobs
from joblib_progress import joblib_progress

def disable_logging():
    logger = logging.getLogger("pcse")
    logger.propagate = False
    logger.setLevel(logging.CRITICAL + 1)  # Plus que CRITICAL pour ne rien logguer
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()


def wof_one_simulation(params_row,
                       wofost_data_path="wofost_data/",
                       output_path="output",
                       # From here only in optimization mode
                       override_params_mode=False,
                       paramset = [],
                       problem={}):
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

    try:
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
        if override_params_mode:
            for name, value in zip(problem["names"], paramset):
                parameters.set_override(name, value)

        wofsim = Wofost73_WLP_CWB(parameters,
                        weatherdata,
                        agromanag_params)
        wofsim.run_till_terminate()
        output = wofsim.get_output()
        dfPP = pd.DataFrame(output).set_index("day")
        # dfPP.to_csv(os.path.join(output_path, f"{params_row['id']}.csv")) #To save une file by simulation
        summary_output = wofsim.get_summary_output()
        crop_cycle = summary_output[0]
        # print(params_row["id"], msg.format(**crop_cycle))
        if override_params_mode:
            return crop_cycle["TAGP"] if params_row["real_crop"] == "Maïs fourrage" else crop_cycle["TWSO"]
        else:
            return params_row["id"], crop_cycle["TWSO"], crop_cycle["TAGP"], crop_cycle["DOH"]
    except Exception as e:
        print(f"Simulation failed for {params_row['id']}: {e}") #TODO: How to handle erros during optimizaiton algorithms?
        if override_params_mode:
            return None
        else:
            return params_row["id"], None, None, None


if __name__ == "__main__":
    with open("src/sims_setup_100_obs.pickle", 'rb') as f:
        sims_data = pickle.load(f)
    simulations = sims_data.to_dict(orient="records")
    print(f"Total simulations to run: {len(simulations)}")
    with joblib_progress(description="Running parallel WOFOST simulations...", total=len(simulations)):
        results = Parallel(n_jobs=60)(
            delayed(wof_one_simulation)(params_row, wofost_data_path="wofost_data/", output_path="output") for params_row in simulations
        )
    answ = pd.DataFrame(results, columns=["id", "TWSO", "TAGP", "DOH"])
    answ.to_pickle("output/base_wofost_test.pkl")
    answ.to_csv("output/base_wofost_test.csv", index=False)
    print(answ)

