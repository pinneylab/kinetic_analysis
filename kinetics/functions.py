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
    # Function to split a list 'l' into 'n' equal-sized chunks.
    # The function yields each chunk as a separate list.
    # It ensures that no chunk is larger than the original list.
    chunk_size = len(l) // n
    remainder = len(l) % n
    start = 0
    for i in range(n):
        if i < remainder:
            end = start + chunk_size + 1
        else:
            end = start + chunk_size
        yield l[start:end]
        start = end

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

### Functions I need to clean up still: ##############################
# These are copied directly from the jupyter notebook for compatibility

def get_initial_slopes(time_arr: np.array, kinetic_data: np.array, plot: bool = False,
                       substrate_concs: [int] = None, 
                       title: str = None, fig_size: (int, int) = (10,10), triage: bool = False):
    
    """
    Fits best initial reaction rate linearly to all kinetic series provided. Optionally plots data for visualization
    and triage.
    
    Args:
        time_arr (np.array): array of assay read times
        kinetic_data (np.array): array of kinetic signal readouts
        plot (bool): flag to turn plotting of fit slopes to scattered data on or off
        substrate_concs ([int]): optional list of substrate concentrations that must be included if plot==True
        title (str): optional title for generated plots
        fig_size ((int, int)): optional dimensions of generated plots
        triage (bool): optional flag to turn on plot "triaging" which separates each substrate's initial rate fitting  
    
    Returns:
        slopes (np.array): array of initial slopes for each sub_array in the kinetic series
        
    """
    if plot and (substrate_concs is None or title is None):
        raise KineticDataException("Plotting initial slopes requires substratce concentration and titles")
    
    slopes = []
    intercepts = []
    scores = []
    
    for data in kinetic_data:
        mask = ~np.isnan(data)
        slope, intercept, score = compute_initial_reaction_slope(time_arr[mask], data[mask])
        slopes.append(slope)
        intercepts.append(intercept)
        scores.append(score)
        
    if plot:
       
        fig = plt.figure(figsize=fig_size) 
        
        for data in kinetic_data:
            plt.scatter(time_arr, data)
        x = np.linspace(min(time_arr), max(time_arr), 1000)
       
        x_tiled = np.tile(x,(len(substrate_concs),1)).T
        plt.plot(x_tiled,
                 x_tiled * np.array(slopes).T + np.array(intercepts).T,
                 linewidth=3, label=[f"{sub_conc:.2f} µM" for sub_conc in substrate_concs])
        plt.title(title)
        plt.xlabel("Time (seconds)")
        plt.ylabel("µM")
        plt.legend() 
    
        if triage:

            fig, axs = plt.subplots(math.ceil(len(sub_concs) / 2), 2, figsize=(15,15))

            for ax, data, slope, intercept, sub_conc in zip(axs.flat, kinetic_data, slopes, intercepts, substrate_concs):
                
                ax.scatter(time_arr[:len(time_arr)//5] , data[:len(time_arr)//5])
                ax.set_title(f"{sub_conc:.2f} µM")
                ax.plot(x[:len(x)//5], x[:len(x)//5] * slope + intercept)
           
    return np.concatenate(slopes)

def fit_and_plot_micheaelis_menten(rep_1_slopes: np.array, rep_2_slopes: np.array, sub_concs: [float], 
                                   e_conc: float, conc_units: str, title: str, background_rates: np.array = None):
    """
    Fits provided initial reaction rates and substrate concentrations to the Michaelis-Menten equation.
    Two replicate must be provided currently (though i will modify this in the future). If you only 
    have one replicate of data, pass it twice.
    
    Args:
        rep_1_slopes (np.array): array of initial slopes for replicate 1
        rep_2_slopes (np.array): array of initial slopes for replicate 2
        sub_concs ([float]): list of substrate concentrations that matches order of samples in each replicate
        e_conc (float): concentration of enzyme in experiment, must be in same units as sub_concs
        conc_units (str): unit name for substrate and enzyme concentration
        title (str): enzyme name/variant/description
        background_rates (np.array): optional arrays of equal dimensions to one replicate, to be subtracted prior to MM fitting
    
    Returns:
        None
        
    """
    avg_slopes = np.nanmean([rep_1_slopes, rep_2_slopes], axis=0)[..., np.newaxis]
   
    if background_rates is not None:
        avg_slopes -= background_rates
    errs = np.nanstd([rep_1_slopes, rep_2_slopes], axis=0) 
    params, _ = curve_fit(mm_model, sub_concs, np.concatenate(avg_slopes))
    
    plt.figure(figsize=(10,6))

    plt.scatter(sub_concs, avg_slopes/e_conc)
    plt.errorbar(sub_concs, np.concatenate(avg_slopes/e_conc), yerr=(errs/e_conc).T, fmt="o", capsize=5)
    x = np.linspace(0, 4000, 1000)
    plt.plot(x, mm_model(x, params[0], params[1]) / e_conc)
    plt.xlabel(f"[S] ({conc_units})")
    plt.ylabel("v ($s^{-1}$)")
    plt.title(title + " kinetics: $k_{cat}$ = " +f"{params[0] / e_conc:.0f}" + " $s^{-1}$ " + f"  $K_m$ = {params[1]:.0f} {conc_units}")
