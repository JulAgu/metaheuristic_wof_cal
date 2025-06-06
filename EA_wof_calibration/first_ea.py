import math
import pandas as pd
import inspyred
import random
import pickle
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import matplotlib.pyplot as plt # TODO: Dont forget to install show the final graph
from wof_tools.wof_ea_interface import set_up_problem, WofostTranslator
from wof_tools.wofost_exec import wof_one_simulation

initial_values = {"wheat": {"TSUM1": 706, "TSUM2": 975},
                  "barley": {"TSUM1": 800, "TSUM2": 750},
                  "fababean": {"TSUM1": 833, "TSUM2": 1351},
                  "maize": {"TSUM1": 600, "TSUM2": 1211},
                  "sunflower": {"TSUM1": 1050, "TSUM2": 1000},
                  "sorghum": {"TSUM1": 730, "TSUM2": 600},
                  "rapeseed": {"TSUM1": 240, "TSUM2": 600},
                  "miller": {"TSUM1": 772, "TSUM2": 483},
                 }
traductor = WofostTranslator()

def init_typical_individual(crop):
    """
    This function initializes a typical individual for a given crop.
    The individual is a dictionary with the crop name as the key and the initial values as the value.
    """
    if crop not in initial_values:
        raise ValueError(f"Crop {crop} not found in initial values.")
    return traductor.wofost_to_genes(list(initial_values[crop].values()))

def naive_generator(random, args):
    initial_values = args.get('initial_values')  # "seed" vector
    noise_std = args.get('noise_std', 0.1)

    # Add Gaussian noise to each value
    return [v + random.gauss(0, noise_std) for v in initial_values]


def naive_parallel_evaluator(candidates, args):
    """
    This function is the naive evaluator. It takes a candidate and returns a score.
    The score is the mean squared error between the candidate and the target.
    """
    def evaluate_one(candidate):
            y_pred = wof_one_simulation(args["row"],
                                        override_params_mode=True,
                                        paramset=traductor.genes_to_wofost(candidate),
                                        problem=args["problem"])
            return abs(args["rdt"] - y_pred)
    with joblib_progress(description ="Parallel process track..."):
         fitness = Parallel(n_jobs=-1)(delayed(evaluate_one)(cand) for cand in candidates) # A vector containing the fitness for each individual in the population
    return fitness


def naive_sequential_evaluator(candidates, args):
    """
    This function is the naive evaluator. It takes a candidate and returns a score.
    The score is the mean squared error between the candidate and the target.
    """
    fitness = []
    for candidate in candidates:
        y_pred = wof_one_simulation(params_row=args["row"],
                                    override_params_mode=True,
                                    paramset=traductor.genes_to_wofost(candidate),
                                    problem=args["problem"])
        mse = abs(args["rdt"] - y_pred)
        fitness.append(mse)
    return fitness


def one_plot_ea(row, problem):
    """
    This function performs the evolutionary algorithm for one plot using the WOFOST model ast the evaluation component.
    It takes a row of the dataframe as input and returns the best individual and its fitness.
    """
    random_number_generator = random.Random()
    random_number_generator.seed(42)

    evolutionary_algorithm = inspyred.ec.EvolutionaryComputation(random_number_generator)
    # and now, we specify every part of the evolutionary algorithm
    evolutionary_algorithm.observer = inspyred.ec.observers.plot_observer
    evolutionary_algorithm.selector = inspyred.ec.selectors.tournament_selection # by default, tournament selection has tau=2 (two individuals), but it can be modified (see below)
    evolutionary_algorithm.variator = [inspyred.ec.variators.uniform_crossover, inspyred.ec.variators.gaussian_mutation] # the genetic operators are put in a list, and executed one after the other
    evolutionary_algorithm.replacer = inspyred.ec.replacers.plus_replacement # "plus" -> "mu+lambda"
    evolutionary_algorithm.terminator = inspyred.ec.terminators.evaluation_termination # the algorithm terminates when a given number of evaluations (see below) is reached
    
    final_population = evolutionary_algorithm.evolve(
        generator = naive_generator, # of course, we need to specify the generator
        evaluator = naive_sequential_evaluator, # and the corresponding evaluator
        pop_size = 100,# 100 # size of the population
        num_selected = 150, # 200 # size of the offspring (children individuals)
        maximize = False, # this is a minimization problem, but inspyred can also manage maximization problem
        max_evaluations = 1000, # 2000? Don't sure maximum number of evaluations before stopping, used by the terminator
        tournament_size = 2, # size of the tournament selection; we need to specify it only if we need it different from 2
        crossover_rate = 1.0, # probability of applying crossover
        mutation_rate = 0.3, # probability of applying mutation
        bounder= inspyred.ec.Bounder(lower_bound=0.0,
                                    upper_bound=1.0),
        # I add a bounder to warranty the values are between 0 and 1, because the genes are normalized
        # all arguments specified below, THAT ARE NOT part of the "evolve" method, will be automatically placed in "args"
        initial_values = init_typical_individual(row["crop"]),
        rdt = row["RealizedYield"],
        row = row,
        problem = problem,
        )
    best_individual = final_population[0]
    return best_individual.candidate, best_individual.fitness


def evaluate_simulation(row):
    candidate, fitness = one_plot_ea(row, problem)
    return row["id"], candidate, fitness


if __name__ == "__main__":
    problem = set_up_problem()
    with open("src/sims_setup.pickle", 'rb') as f:
        sims_data = pickle.load(f)
    simulations = sims_data.to_dict(orient="records")
    with joblib_progress(description="Running parallel WOFOST simulations...", total=len(simulations)):
        results = Parallel(n_jobs=70)(
            delayed(evaluate_simulation)(row) for row in simulations
        )
    # Unpack results
    id_list, candidate_list, fitness_list = zip(*results)
    
    candidate_wofost_list = [traductor.genes_to_wofost(cand) for cand in candidate_list]

    results_df = pd.DataFrame({"ID": id_list,
                               "candidate": candidate_wofost_list,
                               "fitness": fitness_list,})
    results_df.to_pickle("output/wofost_ea_1_results.pkl")
    results_df.to_csv("output/wofost_ea_1_results.csv", index=False)
    print(results_df.head())