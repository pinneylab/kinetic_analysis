import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from scipy.special import lambertw
import matplotlib.cm as cm

class KineticDataException(Exception):
    pass

class ContinuousLinearRegression:
    def __init__(self, X, Y):
        '''
        This class accepts numpy arrays X and Y, and performs a linear regression on them.
        When new data is added, it efficiently updates the linear regression, rather than re-running it.
        Arguments:
            X: numpy array of x-values
            Y: numpy array of y-values
        
        Example:
            X = np.array([1,2,3,4,5])
            Y = np.array([2,4,6,8,10])
            lr = ContinuousLinearRegression(X, Y)
            lr.score()
            >>> 1.0
            lr.update(np.array([6]), np.array([12]))
            lr.score()
            >>> 1.0
        '''
        
        #X, Y are numpy arrays
        self.x = X
        self.y = Y
        self.initialize_internal_vars()
    
    def initialize_internal_vars(self):
        '''
        Initializes the internal variables used to calculate the linear regression.
        This should only performed onced, when the initial X,Y data are provided.
        '''

        self.n = len(self.x)
        
        self.x_sum = np.sum(self.x)         #x_sum
        self.x_squared_sum = np.sum(self.x**2)      #x_squared_sum
        self.determinant = self.n*self.x_squared_sum - self.x_sum**2  #determinant
        self.y_sum = np.sum(self.y)         #y_sum
        self.xy_sum = np.sum(self.x*self.y)  #xy_sum
        self.y_squared_sum = np.sum(self.y**2)      #y_squared_sum
        self.y_avg = self.y_sum/self.n

        #slightly more readable:
        #slope = (n*f-a*e)/D
        #intercept = (b*e-a*f)/D
        self.slope = (self.n*self.xy_sum-self.x_sum*self.y_sum)/self.determinant
        self.intercept = (self.x_squared_sum*self.y_sum-self.x_sum*self.xy_sum)/self.determinant

        self.slope = np.array([self.slope])
    
    def slow_score(self):
        '''
        Calculates the R2 of the current fit of the data, using the slow method.
        '''

        y_pred = self.slope*self.x + self.intercept
        ss_res = np.sum((self.y - y_pred) ** 2)
        ss_tot = np.sum((self.y - self.y_avg) ** 2)
        self.r_squared = 1 - (ss_res / ss_tot)
        return self.r_squared

    def update(self, new_x, new_y):
        '''
        Updates the internal variables used to calculate the linear regression.
        Then, calculates the new slope and intercept.
        '''
        self.x = np.append(self.x, new_x)
        self.y = np.append(self.y, new_y)

        self.n = len(self.x)
        
        self.x_sum = self.x_sum + np.sum(new_x)
        self.x_squared_sum = self.x_squared_sum + np.sum(new_x**2)
        self.determinant = self.n*self.x_squared_sum - self.x_sum**2
        self.y_sum = self.y_sum + np.sum(new_y)
        self.xy_sum = self.xy_sum + np.sum(new_x*new_y)
        self.y_squared_sum = self.y_squared_sum + np.sum(new_y**2)
        self.y_avg = self.y_sum/self.n

        #calculate slope and intercept
        self.slope = (self.n*self.xy_sum-self.x_sum*self.y_sum)/self.determinant
        self.intercept = (self.x_squared_sum*self.y_sum-self.x_sum*self.xy_sum)/self.determinant

        self.slope = np.array([self.slope])
    
    def score(self):
        '''
        Calculates the R2 of the current fit of the data.
        Performs this using the continuously updated variables we establish in the update function.
        '''
        SS_res = self.y_squared_sum - 2*self.slope*self.xy_sum - 2*self.intercept*self.y_sum + self.slope**2*self.x_squared_sum + 2*self.slope*self.intercept*self.x_sum + self.n*self.intercept**2
        SS_tot = self.y_squared_sum - 2*self.y_avg*self.y_sum + self.n*self.y_avg**2
        self.r_squared = 1 - SS_res/SS_tot
        return self.r_squared

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

def compute_exponential_fit(time_arr: np.array,
                            signal_arr: np.array,
                            y_intercept: float,
                            add_linear: bool = False,
                            plot: bool = False):
    """
    Fits an exponential function to the data. The function is of the form:
    [P]_t = [S]_0 * (1 - e^(-kt))
    where [P]_t is the product concentration at time t, [S]_0 is the initial substrate concentration,
    and k is the rate constant.
    The function returns the rate constant, k.
        Args:
        time_arr (np.array): array of assay read times
        signal_arr (np.array): array of kinetic signal readouts
        y_intercept (float): y-intercept of the linear fit
        add_linear (bool): whether to add a linear term to the exponential fit
    
    Returns:
        list of parameters:
            S0 (float): initial substrate concentration
            k (float): rate constant
            c (float): y-intercept
            evap_rate (float): rate of evaporation and photobleaching
    """
    # define the exponential function to fit
    def exp_func(t, S0, k, e=0):
        #S0 is our initial substrate concentration
        #k is our forward reaction rate (assumed to be >> reverse rate)
        #c is our y-intercept. If the reaction began while we were recording, this would be 0. However, we always have a delay.
        #evap_rate is the rate of evaporation AND photobleaching. Assumed to be constant over the course of the reaction.

        #return S0 * (1 - np.exp(-k * t)) + c - evap_rate * t
        return S0 * (1 - np.exp(-k * t)) + y_intercept + e*t

    # fit the exponential function to the data
    if add_linear: #if we're adding a linear term, we need to provide an initial guess for the slope
        p0=(100,0,0)
    else:
        p0=(100,0)
    popt, pcov = curve_fit(exp_func, time_arr, signal_arr, p0=p0, method='dogbox') #should we provide the S0, instead of fitting it?

    if plot:
        # plot the original data and the fitted function
        plt.plot(time_arr, signal_arr, 'b-', label='data')
        plt.plot(time_arr, exp_func(time_arr, *popt), 'r-', label='fit')
        plt.title('Exponential fit')
        plt.legend()
        plt.show()

    # return the fit parameters:
    return (*popt, pcov)

def compute_initial_reaction_slope_fast(time_arr: np.array, 
                                   signal_arr: np.array,
                                  min_included_percent: int = 2): 
    """
    Determines best linear fit to initial slope of data, by fitting regressions to 
    all percentiles of the data--greater than some defined minimum--anchored at the origin.
    This new method performs one linear regression and updates it continuously as more data is added, rather than re-running each time.
    
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
    
    #does this make sense? 
    min_inclusion = max(len(perc_times) * min_included_percent // 100, MIN_INCLUDED_DATA_POINTS)
    
    #Get initial fit:
    x = np.concatenate(perc_times[:min_inclusion])
    y = np.concatenate(perc_concs[:min_inclusion])

    reg = ContinuousLinearRegression(x, y)
    r2 = reg.score()
    scores.append(r2)
    slopes.append(reg.slope)
    intercepts.append(reg.intercept)

    #Update fit as more data is added:
    for i in range(min_inclusion, len(perc_times)):
        #add to the end x,y
        new_x = perc_times[i]
        new_y = perc_concs[i]

        reg.update(new_x, new_y)
        r2 = reg.score()
        scores.append(r2)
        slopes.append(reg.slope)
        intercepts.append(reg.intercept)
    
    if len(scores) == 0 and min_included_percent == 100:
        print('Performing regression on all data points')
        reg = LinearRegression().fit(np.array(perc_times).reshape(-1,1), perc_concs)
        curr_score = reg.score(np.array(perc_times).reshape(-1,1), perc_concs)
        return reg.coef_.item(), reg.intercept_.item(), curr_score

    # The fit with the highest R-squared value is selected as the best fit.
    max_r2_idx = np.argmax(scores)
    
    if scores[max_r2_idx] < 0.9:
        return np.array([np.nan]),np.nan, np.array([np.nan])

    #old code expects an array of arrays for slopes. This should be fixed eventually.
    return slopes[max_r2_idx], intercepts[max_r2_idx], scores[max_r2_idx]

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
    
    #for i in range(len(perc_times), min_inclusion, -1):
    for i in range(min_inclusion, len(perc_times), 1):
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
        return np.array([np.nan]), np.nan, np.array([np.nan])
    
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
                       title: str = None, fig_size: (int, int) = (10,10), triage: bool = False, mode="linear_fast"):
    
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
        mode (str): modes for fitting initial slopes.
            "linear": finds a linear fit to the initial slope of the data
            "linear_fast": the same method as above, but much faster. This is the default.
            "exponential": fits an exponential function to all of the data, and calculates the slope from the rate constant
            "exponential_linear": the same as above, but the exponential function has a linear term added to account for evaporation and photobleaching.
    
    Returns:
        slopes (np.array): array of initial slopes for each sub_array in the kinetic series
        scores (np.array): array of R^2 values for each fit (coefficient of determination)
        
    """
    if plot and (substrate_concs is None or title is None):
        raise KineticDataException("Plotting initial slopes requires substratce concentration and titles")
    
    slopes = []
    intercepts = []
    scores = []
    
    if mode in ("linear", "linear_fast"):
        for data in kinetic_data:
            mask = ~np.isnan(data)
            if mode == "linear":
                slope, intercept, score = compute_initial_reaction_slope(time_arr[mask], data[mask])
            else:
                slope, intercept, score = compute_initial_reaction_slope_fast(time_arr[mask], data[mask])
            #print(slope, intercept, score)
            #compute_exponential_fit(time_arr[mask], data[mask])
            slopes.append(slope)
            intercepts.append(intercept)
            scores.append(score)
    elif mode in ("exponential", "exponential_linear"):
        for data in kinetic_data:
            #Do I need a NaN mask here?

            #We want to exclude data after the reaction finishes. We first find the index where the smoothed data is at maximum (max_index)
            window_size = 10
            data_smooth = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
            #make sure we're not losing beginning and end of data
            data_smooth = np.append(data_smooth, data[-window_size+1:])
            max_index = np.argmax(data_smooth)

            #we truncate the data to this point:
            data_trunc = data[:max_index]
            time_trunc = time_arr[:max_index]

            #calculate the y intercept, and fit:
            y_intercept = data_trunc[0] #the cheap way to do this is to just use the first data point as the y-intercept
            if mode=="exponential":
                S0, k, pcov= compute_exponential_fit(time_trunc, data_trunc, y_intercept)
            else:
                S0, k, e, pcov= compute_exponential_fit(time_trunc, data_trunc, y_intercept, add_linear=True)
            
            #calculate the slope from substrate conc, rate constant:
            V0 = S0*k

            slopes.append(np.array([V0]))
            intercepts.append(y_intercept)
            scores.append(pcov)
    else:
        raise KineticDataException("Mode must be either 'linear', 'exponential', or 'exponential_linear'")
        
    if plot:
       
        #we'll plot the points in the first 20% of time, for easy visualization.
        max_time = max(time_arr)
        twenty_percent_time = max_time * 0.2
        mask = time_arr < twenty_percent_time

        fig = plt.figure(figsize=fig_size) 
        
        for data in kinetic_data:
            
            plt.scatter(time_arr[mask], data[mask])
            #plt.scatter(time_arr, data)
        x = np.linspace(min(time_arr[mask]), max(time_arr[mask]), 1000)

        x_tiled = np.tile(x,(len(substrate_concs),1)).T
        plt.plot(x_tiled,
                 x_tiled * np.array(slopes).T + np.array(intercepts).T,
                 linewidth=3, label=[f"{sub_conc:.2f} µM" for sub_conc in substrate_concs])
        plt.title(title)
        plt.xlabel("Time (seconds)")
        plt.ylabel("µM")
        plt.legend() 
    
        if triage:

            fig, axs = plt.subplots(math.ceil(len(substrate_concs) / 2), 2, figsize=(15,15))

            for ax, data, slope, intercept, sub_conc in zip(axs.flat, kinetic_data, slopes, intercepts, substrate_concs):
                
                ax.scatter(time_arr[:len(time_arr)//5] , data[:len(time_arr)//5])
                ax.set_title(f"{sub_conc:.2f} µM")
                ax.plot(x[:len(x)//5], x[:len(x)//5] * slope + intercept)
    
    #Check if any slopes are NaN. If so, output descriptive warning:
    for i in range(len(slopes)):
        if np.isnan(slopes[i]):
            # print(f"WARNING: slope for {substrate_concs[i]} µM is NaN.\n\ 
            #         This is likely due to scipy failing to fit a linear model to the data.\n")
            print(f"WARNING: slope for {substrate_concs[i]} µM is NaN.\n This is likely due to scipy failing to fit a linear model to the data.")
    
    #print(slopes)
    return np.concatenate(slopes), np.concatenate(scores)

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
    max_substrate = max(sub_concs)
    x = np.linspace(0, max_substrate, 1000)
    plt.plot(x, mm_model(x, params[0], params[1]) / e_conc)
    plt.xlabel(f"[S] ({conc_units})")
    plt.ylabel("v ($s^{-1}$)")
    plt.title(title + " kinetics: $k_{cat}$ = " +f"{params[0] / e_conc:.0f}" + " $s^{-1}$ " + f"  $K_m$ = {params[1]:.0f} {conc_units}")

def fit_michaelis_mentin(rep_1_slopes: np.array, rep_2_slopes: np.array, sub_concs: [float], 
                                   e_conc: float, conc_units: str, title: str, background_rates: np.array = None):
    """
    Copy of fit_and_plot_micheaelis_menten without plotting functionality, and returning kinetic parameters.
    In the next release, we should combine the two with a plotting option.
    
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
        k_cat (float): turnover rate of enzyme
        K_m (float): Michaelis constant of enzyme
        perr (np.array): standard deviation of errors of the parameters
        
    """
    avg_slopes = np.nanmean([rep_1_slopes, rep_2_slopes], axis=0)[..., np.newaxis]
   
    if background_rates is not None:
        avg_slopes -= background_rates
    errs = np.nanstd([rep_1_slopes, rep_2_slopes], axis=0) 
    params, pcov = curve_fit(mm_model, sub_concs, np.concatenate(avg_slopes))
    
    #compute standard deviation of errors of the parameters:
    perr = np.sqrt(np.diag(pcov))
    
    return params[0] / e_conc, params[1], perr

def fit_michaelis_mentin_lambert_omega(time_arr: np.array, kinetic_data: np.array, plot: bool = False,
                       substrate_concs: [int] = None, protein_conc: float = None,
                       title: str = None, fig_size: (int, int) = (10,10), triage: bool = False, model: str = 'integrated_MM'):
    
    """
    Calculates the Michaelis-Menten parameters for the given kinetic data using the Lambert-W function.
    This function is described in "Parameter estimation using a direct solution of the integrated Michaelis-Menten equation", Gouddar et. al.
    Args:
        time_arr (np.array): array of assay read times
        kinetic_data (np.array): array of kinetic signal readouts
        plot (bool): flag to turn plotting of fit slopes to scattered data on or off
        substrate_concs ([int]): optional list of substrate concentrations that must be included if plot==True
        title (str): optional title for generated plots
        fig_size ((int, int)): optional dimensions of generated plots
        triage (bool): optional flag to turn on plot "triaging" which separates each substrate's initial rate fitting  
    """
    
    def integrated_michaelis_menten_equation(time: np.ndarray, Km: float, Vmax: float, s0: float):
        z = (s0 / Km) * np.exp(np.subtract(s0, Vmax * time) / Km)
        product_concens = s0 - (Km * np.real(lambertw(z))) # assuming we only have to use the principal real branch
        return product_concens

    def fitting_function_background(time: np.ndarray, beta: float):
        return beta * time

    def fitting_function_signal(time: np.ndarray, F0: float, FF: float, Km:float, Vmax: float, s0: float):
        m = (FF - F0) / s0
        fluorescence = m * integrated_michaelis_menten_equation(time, Km, Vmax, s0) + F0
        return fluorescence    
    
    F0s, FFs, Kms, kcats, betas, pconvs = [], [], [], [], [], []
    for i, data in enumerate(kinetic_data):
        mask = ~np.isnan(data)
        time_arr_masked = time_arr[mask]
        data_masked = data[mask]
        substrate_conc = substrate_concs[i]

        if model == 'integrated_MM':
            wrapper = lambda time, F0, FF, Km, Vmax: fitting_function_signal(time, F0, FF, Km, Vmax, substrate_conc)
            lb, ub = np.array([0, 0, 0, 0]), np.array([np.inf, np.inf, np.inf, np.inf])
            initial_guess = np.array([data_masked[0], data_masked[-1], 1, 0.001])
            popt, pconv = curve_fit(wrapper, time_arr_masked, data_masked, bounds=(lb, ub), p0=initial_guess)

            # unpack and organize
            F0, FF, Km, Vmax = popt 
            F0s.append(F0)
            FFs.append(FF)
            Kms.append(Km)
            kcats.append(Vmax / (protein_conc))
            betas.append('NA')
            pconvs.append(pconv)

        elif model == 'integrated_MM_with_background':
            wrapper = lambda time, F0, FF, Km, Vmax, beta: fitting_function_signal(time, F0, FF, Km, Vmax, substrate_conc) + fitting_function_background(time, beta)
            lb, ub = np.array([0, 0, 0, 0, -np.inf]), np.array([np.inf, np.inf, np.inf, np.inf, np.inf])
            initial_guess = np.array([data_masked[0], data_masked[-1], 1, 0.001, 1])
            popt, pconv = curve_fit(wrapper, time_arr_masked, data_masked, bounds=(lb, ub), p0=initial_guess)

            # unpack and organize
            F0, FF, Km, Vmax, beta = popt 
            F0s.append(F0)
            FFs.append(FF)
            Kms.append(Km)
            kcats.append(Vmax / (protein_conc))
            betas.append(beta)
            pconvs.append(pconv)

        else:
            print(f'ERROR: f{model} not recognized as a valid model to fit to. Valid models include "integrated_MM" or "integrated_MM_with_background".')
            return
        
    if plot:
        #Plot fit of Kcat and Km
        fig, axs = plt.subplots(1, 2, figsize=(15,5))
        axs[0].scatter(substrate_concs, kcats)
        axs[0].set_xlabel('Substrate Concentration (uM)')
        axs[0].set_ylabel('Kcat (1/s)')
        axs[0].set_title(title)
        axs[1].scatter(substrate_concs, Kms)
        axs[1].set_xlabel('Substrate Concentration (uM)')
        axs[1].set_ylabel('Km (uM)')
        axs[1].set_title(title)