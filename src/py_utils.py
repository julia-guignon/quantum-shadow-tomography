import numpy as np
from src.classical_shadows import *


def replace_in_list(lst, a, b):
    '''
        Replaces a by b in a list lst. Returns a new list.
    '''
    return list(map(lambda x: x if x!=a else b, lst))

# def flatten_list(lst):
#     '''
#         Converts list of list into a list by flattening it.
#     '''
#     return sum(lst, [])

def get_values_dict_of_dict(dct):
    '''
        Returns a single list of all the values in a dict of dict.
    '''
    return sum([[v for v in el.values()] for el in list(dct.values())], [])


def bootstrap_confidence_interval(data, size, K, iterations=1000):
    """
    Bootstrap the 95% confidence interval for the mean of the data.
    
    Parameters:
    - data: An array of data
    - iterations: The number of bootstrap samples to generate
    
    Returns:
    - A tuple representing the lower and upper bounds of the 95% confidence interval
    """
    means = np.zeros(iterations)
    
    for i in range(iterations):
        bootstrap_sample = np.random.choice(data, size=size, replace=True)
        means[i] = median_of_means_fidelity(bootstrap_sample, K)
        
    lower_bound = np.percentile(means, 2.5)
    upper_bound = np.percentile(means, 97.5)
    mu = np.mean(means)
    return [mu, mu-lower_bound, upper_bound-mu]
    