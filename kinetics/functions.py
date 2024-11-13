import numpy as np
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit, minimize
from scipy.special import lambertw


class BindingModel:
    def __init__(self):
        pass 

    def __call__(self):
        pass 

    @staticmethod 
    def binding_model(x: np.ndarray, kd: float, rmax: float):
        pass

    def fit(x, y, fixed_rmax: float = None):

        def objective(p):
            kd, rmax = p
            yhat = BindingModel.binding_model(x, kd, rmax)
            pass 

        if fixed_rmax:
            objective = lambda kd: objective(np.array([kd, fixed_rmax]))

        result = minimize(objective, x, y, p0=np.array([1,1]))

        pass 


class SingleExponentialModel:
    """ 
    Class for fitting exponential models to progress curve data.
    """

    def __init__(self):
        pass

    def __call__(self, x: np.ndarray, parameters: dict):
        k, span, plateau = parameters['k'], parameters['span'], parameters['plateau']
        return SingleExponentialModel.single_exponential(x, k, span, plateau)
    
    @staticmethod
    def single_exponential(x: np.ndarray, k: float, span: float, plateau: float):
        return (span * np.exp(k * x)) + plateau
    
    def fit(self, x: np.ndarray, y: np.ndarray):

        # compute initial guesses
        plateau0 = y[-1]
        span0 = y[0] - plateau0
        
        if y[-1] > y[0]:
            mask = (y < plateau0) & (x > 0)
        else:
            mask = (y > plateau0) & (x > 0)

        k0 = (np.log((y[mask] - plateau0) / span0) / x[mask]).mean()
        p0 = np.array([k0, span0, plateau0])

        # compute fit
        try:
            popt, pcov = curve_fit(SingleExponentialModel.single_exponential, x, y, p0=p0)
            parameters = {
                'k': popt[0],
                'span': popt[1],
                'plateau': popt[2],
                'pcov': pcov
            }
        except:
            parameters = {
                'k': np.nan,
                'span': np.nan,
                'plateau': np.nan,
                'pcov': np.nan
            }

        return parameters


class LinearModel:
    """ 
    Wrapper class over scipy.stats.linregress for fitting lines to data.
    """

    def __init__(self):
        pass

    def __call__(self, x: np.ndarray, parameters: dict):
        slope, intercept = parameters['slope'], parameters['intercept']
        return (x * slope) + intercept

    def fit(self, x: np.ndarray, y: np.ndarray):
        result = linregress(x, y)
        parameters = {
            'slope': result.slope,
            'intercept': result.intercept,
            'r2': result.rvalue
        }
        return parameters


class InitialRateModel(LinearModel):
    """
    Class for fitting initial rates to progress curve data.
    """

    def __init__(self):
        pass

    def fit(self, x: np.ndarray, y: np.ndarray, min_included_percent: float = 2, exhaustive: bool = False):
        """ 
        Wrapper function over pre-existing code for fitting initial rates.
        """

        if exhaustive:
            pass

        else:
            slope, intercept, r2 = compute_initial_reaction_slope_fast(x, y, min_included_percent=min_included_percent)
            parameters = {
                'slope': slope[0],
                'intercept': intercept,
                'r2': r2[0]
            }

        return parameters
    

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


class MichaelisMentenModel:
    def __init__(self):
        pass

    def __call__(self, x: np.ndarray, parameters: dict):
        kcat, km = parameters['kcat'], parameters['km']
        return MichaelisMentenModel.michaelis_menten_model(x, kcat, km)

    @staticmethod
    def michaelis_menten_model(x: np.ndarray, kcat: float, km: float):
        return x * kcat / (km + x)

    def fit(self, x: np.ndarray, y: np.ndarray):

        try: 
            popt, pcov = curve_fit(MichaelisMentenModel.michaelis_menten_model, x, y)
            parameters = {
                'kcat': popt[0],
                'km': popt[1],
                'pcov': pcov
            }
        except:
            parameters = {
                'kcat': np.nan,
                'km': np.nan,
                'pcov': np.nan
            }
        return parameters


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

    #old code expects an array of arrays for slopes. This should be fixed eventually.
    return slopes[max_r2_idx], intercepts[max_r2_idx], scores[max_r2_idx]


### Functions I need to clean up still: ##############################
# These are copied directly from the jupyter notebook for compatibility


def fit_michaelis_menten_lambert_omega(time_arr: np.array, kinetic_data: np.array, plot: bool = False,
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