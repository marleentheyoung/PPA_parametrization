"""
This file contains the code needed to generate a variety of parameter configurations. Next to this,
the functions get_min_min() and get_max_min() are necessary for the heatmap visualizations.

"""

import pandas as pd

FUNCTIONS = {'easom':-1, 'goldstein_price':3, 'martin_gaddy':0, 'six_hump_camel':-1.031628453489877 , 'branin':0.397887}
MODES = ['average_best', 'median_best', 'min_min', 'optimum']

def config_generator(n_range=[1,11], m_range=[1, 41]):
    """
    Generates a variety of parameter configurations for parameters
    popSize and max_runners

    args:
        n_range: range of n_max
        m_range: range of m (popSize)

    returns:
        A list of configuration dicts
    """
    configs = []
    config = {}

    for n_max in range(n_range[0], n_range[1]):
        for m in range(m_range[0], m_range[1]):
            config["m"] = m
            config["max_runners"] = n_max
            configs.append({f"n_max={n_max}_m={m}":config})
    return configs

def get_max_min(functions):
    """
    Returns the highest (worst) found median optimum over all 
    runs of all tested functions.

    args:
        n_range: range of n_max
        m_range: range of m (popSize)

    returns:
        max min 
    """
    max_mins = []
    for function in functions:
        filename = f"../data/cross/median_best/relation_n_max_to_m_{function}.csv"
        results = pd.read_csv(filename)
        max_min = results['median_best'].max() - FUNCTIONS[function]
        max_mins.append(max_min)
    return max(max_mins)

def get_min_min(functions):
    """
    Returns the best found (min min) median optimum over all 
    runs of all test functions.

    args:
        n_range: range of n_max
        m_range: range of m (popSize)

    returns:
        min min (float)
    """
    min_mins = []
    for function in functions:
        filename = f"../data/cross/median_best/relation_n_max_to_m_{function}.csv"
        results = pd.read_csv(filename)
        min_min = results['median_best'].min() - FUNCTIONS[function]
        min_mins.append(min_min)
    return min(min_mins)

def get_statistics(n_range, m_range, function, optimum):
    """
    Calculates mean and sigma of values within a range of parameter 
    configurations.

    args:
        n_range: range of n_max
        m_range: range of m (popSize)
        function: name of function
        optimum: global optimum of specified function

    returns:
        tuple (mean, sigma)
    """
    min_mins = []
    filename = f"../data/cross/median_best/relation_n_max_to_m_{function}.csv"
    results = pd.read_csv(filename)
    results = results[(results['n_max'] >= n_range[0]) & (results['n_max'] <= n_range[1])]
    results = results[(results['m'] >= m_range[0]) & (results['m'] <= m_range[1])]
    results = results.subtract(optimum)
    return (results['median_best'].mean(), results['median_best'].std())