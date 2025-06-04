"""
This script has as objective to provide the communication tools between the WOFOST model and the evolutionary algorithm.
TODO: The first test is going to be to learn better TSUM1 and TSUM2. Then we will add the other parameters.
"""
def get_ranges(parameter):
    """
    Get the min and max values of a parameter from the ranges dictionary.
    """
    #TODO: In the future we will have to add the other parameters as a function of a sensitivity analysis.
    RANGES = {"TSUM1": (100, 1000),
              "TSUM2": (800, 2000),
              }
    if parameter in RANGES:
        return RANGES[parameter]
    else:
        raise ValueError(f"Parameter {parameter} not found in ranges.")

def wofost_to_genes(wof_value, min, max):
    return (wof_value - min) / (max - min)

def genes_to_wofost(gen_value, min, max):
    return gen_value * (max - min) + min

