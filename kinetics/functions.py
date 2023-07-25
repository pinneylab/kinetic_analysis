import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import matplotlib.cm as cm

class KineticDataException(Exception):
    pass

def divide_chunks(l, n):
    # Function to split a list 'l' into 'n' roughly equal-sized chunks.
    # The function yields each chunk as a separate list.
    # It ensures that no chunk is larger than the original list.
    n = math.ceil(len(l) / float(n))
    for i in range(0, len(l), n):
        yield l[i:i + n]

def mm_model(x, v_max, Km):
    # Michaelis-Menten equation to model enzyme kinetics.
    # 'x': substrate concentration, 'v_max': maximum reaction rate, 'Km': Michaelis constant.
    return v_max * x / (x + Km)


def compute_initial_reaction_slope(time_arr: np.array, 
                                   signal_arr: np.array,
                                  min_included_percent: int = 2): 
    """
    Determines best linear fit to initial slope of data, by fitting regressions to 
    all percentiles of the data--greater than some defined minimum--anchored at the origin.
    
    Args:
        time_arr (np.array): array of assay read times
        signal_arr (np.array): array of kinetic signal readouts
        min_included_percent (int) = 5: minimum percent of data to be included 
    
    Returns:
        slope, intercept, score ((float, float, float)): fit parameters of best fit
        
    """
    # Need to triage further for different definitions of minimum
    MIN_INCLUDED_DATA_POINTS = 3
    
    perc_concs = list(divide_chunks(signal_arr, 100))
    perc_times = list(divide_chunks(time_arr, 100))
 
    scores = []
    slopes = [] 
    intercepts = []
    
    min_inclusion = max(len(perc_times) * min_included_percent // 100, MIN_INCLUDED_DATA_POINTS)
    
    for i in range(len(perc_times), min_inclusion, -1):
        curr_times = np.concatenate(perc_times[:i])
        curr_concs = np.concatenate(perc_concs[:i])
        reg = LinearRegression().fit(np.array(curr_times).reshape(-1,1), curr_concs)
        curr_score = reg.score(np.array(curr_times).reshape(-1,1), curr_concs)
        scores.append(curr_score)
        slopes.append(reg.coef_)
        intercepts.append(reg.intercept_)
    
    if len(scores) == 0 and min_included_percent == 100:
        reg = LinearRegression().fit(np.array(perc_times).reshape(-1,1), perc_concs)
        curr_score = reg.score(np.array(perc_times).reshape(-1,1), perc_concs)
        return reg.coef_.item(), reg.intercept_.item(), curr_score

    # The fit with the highest R-squared value is selected as the best fit.
    max_r2_idx = np.argmax(scores)
    
    if scores[max_r2_idx] < 0.9:
        return np.array([np.nan]), np.array([np.nan]), np.array([np.nan])
    
    return slopes[max_r2_idx], intercepts[max_r2_idx], scores[max_r2_idx]


def get_scoop_filter_index(times: list, data: list):
    # Find the index to filter data to remove the initial scooping artifact in kinetic measurements.
    # The scooping artifact is the region where the curve decreases at the beginning of the reaction.
    rate_of_change = np.array(data[1:]) - np.array(data[:-1])
    try:
        last_negative_index = np.where(rate_of_change < 0)[0][-1] + 1
    except IndexError:
        last_negative_index = 0
    duration = max(times) - min(times)
    max_time_cutoff = np.where(np.array(times) < (duration / 3))[0][-1]

    return min(last_negative_index, max_time_cutoff)

   
def fit_kinetics_linear(row):
    # Fit linear models to the initial slope of the kinetic data for each row (reaction) in the dataset.
    # The function uses 'compute_initial_reaction_slope' to compute the initial slope fit,
    # and 'get_scoop_filter_index' to determine the index to filter out the scooping artifact.
    # It returns slope, intercept, and R-squared values for both the entire reaction and the filtered part.
    try:
        slope, intercept, score = compute_initial_reaction_slope(row['time_s'],row['chamber_product_concs'])
    except ValueError:
        
        slope = np.array([np.nan])
        intercept = np.array([np.nan])
        score = np.array([np.nan])
            
    filter_scoop_index = get_scoop_filter_index(row['time_s'],row['chamber_product_concs'])
    try:
        filter_slope, filter_intercept, filter_score = compute_initial_reaction_slope(row['time_s'][filter_scoop_index:],row['chamber_product_concs'][filter_scoop_index:])
    except ValueError:
        
        filter_slope = np.array([np.nan])
        filter_intercept = np.array([np.nan])
        filter_score = np.array([np.nan])
    return slope.item(), intercept.item(), score.item()**2, filter_slope.item(), filter_intercept.item(), filter_score.item()**2
    