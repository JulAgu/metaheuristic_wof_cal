"""
This script has as objective to provide the communication tools between the WOFOST model and the evolutionary algorithm.
TODO: The first test is going to be to learn better TSUM1 and TSUM2. Then we will add the other parameters.
"""

def set_up_problem():
    """
    Creates the problem dictionaries for sensitivity analysis.
    """
    RANGES_VAR = {
                  "TSUM1": (100, 2000),
                  "TSUM2": (100, 2000),
                #   "SPAN":  (10, 70),
                #   "CFET": (0.1, 1.0),
                #   "CVL": (0.1, 1.0),
                #   "CVO": (0.1, 1.0),
                #   "CVR": (0.1, 1.0),
                #   "CVS": (0.1, 1.0),
                #   "TBASE": (0, 15),
                #   "TBASEM": (0, 15),
                #   "VERNBASE": (5, 15),
                #   "RDI": (9, 11),
                #   "RDMCR": (60, 300),
                #   "RGRLAI": (0.001, 0.8),
                  }

    RANGES_SOIL = {
                #    "K0": (10, 100),
                #    "SOPE": (0.2, 15),
                #    "KSUB": (0.1, 30),
                #    "RDMSOL": (90, 150),
                   }
    
    problem = {
        "num_vars": len(RANGES_VAR) + len(RANGES_SOIL),
        "names": list(RANGES_VAR.keys()) + list(RANGES_SOIL.keys()),
        "bounds": [list(bounds) for bounds in RANGES_VAR.values()] + [list(bounds) for bounds in RANGES_SOIL.values()]
        }
 
    return problem

class WofostTranslator:
    """
    This class provides methods to translate WOFOST model parameters to genes and vice versa.
    """
    def __init__(self):
        self.ranges_dic = {"TSUM1": (100, 2000),
                           "TSUM2": (800, 2000),
                          }
        self.ranges = [i for i in self.ranges_dic.values()]
        
    def print_available_parameters(self):
        """
        Print the available parameters and their ranges.
        """
        print("Available parameters and their ranges:")
        for parameter, (min_val, max_val) in self.ranges_dic.items():
            print(f"{parameter}: {min_val} - {max_val}")
    
    def get_one_range(self, parameter):
        """
        Get the min and max values of a parameter from the ranges dictionary.
        """
        #TODO: In the future we will have to add the other parameters as a function of a sensitivity analysis.
        if parameter in self.ranges_dic:
            return self.ranges_dic[parameter]
        else:
            raise ValueError(f"Parameter {parameter} not found in ranges.")

    def wofost_to_genes(self, wof_values):
        list_of_genes = [(w_v - min) / (max - min) for w_v, (min, max) in zip(wof_values, self.ranges)]
        return list_of_genes

    def genes_to_wofost(self, gen_values):
        list_of_wofost = [g_v * (max - min) + min for g_v, (min, max) in zip(gen_values, self.ranges)]
        return list_of_wofost

