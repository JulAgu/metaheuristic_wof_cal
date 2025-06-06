"""
Reusing the logic herited from wofost_plot_ea, this script is designed to run a random search baseline for the WOFOST model.
"""
import random
import pickle
import pandas as pd
import tqdm as tqdm
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from tqdm import tqdm
from wof_tools.wof_ea_interface import set_up_problem, WofostTranslator
from wof_tools.wofost_exec import wof_one_simulation

#TODO: This dictionary is present in multiple scripts
initial_values = {"wheat": {"TSUM1": 706, "TSUM2": 975},
                  "barley": {"TSUM1": 800, "TSUM2": 750},
                  "fababean": {"TSUM1": 833, "TSUM2": 1351},
                  "maize": {"TSUM1": 600, "TSUM2": 1211},
                  "sunflower": {"TSUM1": 1050, "TSUM2": 1000},
                  "sorghum": {"TSUM1": 730, "TSUM2": 600},
                  "rapeseed": {"TSUM1": 240, "TSUM2": 600},
                  "millet": {"TSUM1": 772, "TSUM2": 483},
                   }
traductor = WofostTranslator()

def random_generator(ranges):
    """
    This function generates a random individual based on the initial values and noise standard deviation.
    """
    return [random.uniform(min_val, max_val) for min_val, max_val in ranges]


def random_searcher(row, problem, n_iterations=1000):
    """
    This function performs a random search for the WOFOST model.
    It generates a random individual and evaluates it using the WOFOST model.
    """
    def evaluate_one(candidate):
        """
        This function evaluates a single candidate by running the WOFOST model and returning the fitness.
        """
        y_pred = wof_one_simulation(params_row=row,
                                    override_params_mode=True,
                                    paramset=candidate,
                                    problem=problem)
        fitness = abs(row["RealizedYield"] - y_pred)
        return candidate, fitness

    crop = row["crop"]
    candidates = [list(initial_values[crop].values())]
    ranges = traductor.ranges
    next_candidates = [random_generator(ranges) for _ in range(n_iterations-1)]
    candidates.extend(next_candidates)

    # with joblib_progress(description ="Parallel process track..."):
    #TODO: Try the parallelization over the plots not the candidates. This improves the performance?
    results = Parallel(n_jobs=60)(delayed(evaluate_one)(cand) for cand in candidates)
    results.sort(key=lambda x: x[1])
    best_candidate, best_fitness = results[0]
    return best_candidate, best_fitness

if __name__ == "__main__":
    problem = set_up_problem()
    with open("src/sims_setup.pickle", 'rb') as f:
        sims_data = pickle.load(f)
    simulations = sims_data.to_dict(orient="records")
    id_list = []
    candidate_list = []
    candidate_wofost_list = []
    fitness_list = []
    for row in tqdm(simulations):
        candidate, fitness = random_searcher(row, problem, n_iterations=100)
        # print(candidate, fitness)
        id_list.append(row["id"])
        candidate_list.append(candidate)
        fitness_list.append(fitness)

    results_df = pd.DataFrame({"id": id_list,
                               "candidate": candidate_list,
                               "fitness": fitness_list,})
    results_df.to_pickle("output/random_search_results.pkl")
    results_df.to_csv("output/random_search_results.csv", index=False)
    print(results_df.head())