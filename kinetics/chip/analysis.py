### NF: Eventually, I'd like to move this into HTBAM_analysis repo. However, since our repos are private and this requires importing kinetics, it's here for now.
###     Currently, HTBAM_analysis is only Fordyce code.

import numpy as np
from matplotlib import pyplot as plt
import scipy

def new_analysis(db, run_name, analysis_name, analysis_type):
    ''' This function performs a new analysis on the data in the database.
        Inputs:
            db: a Database object
            run_name: the name of the run to analyze
            analysis_name: the name of the analysis to perform
            analysis_type: the type of analysis to perform
                supported types: 'linear_regression'
        Outputs:
            None
        Side effects:
            Saves the analysis to the database.
    '''

    if analysis_name == 'linear_regression':
        _linear_regression_analysis(db, run_name)
    else:
        raise ValueError(f'Analysis name {analysis_name} not recognized.')


def _linear_regression_analysis(db, run_name):
    ''' This function performs a linear regression analysis on the data in the database.
        Inputs:
            db: a Database object
        Outputs:
            None
        Side effects:
            Saves the analysis to the database.
    '''
    #Here, we collect our data into numpy arrays
    chamber_idxs, luminance_data, conc_data, _ = db.get_run_data(run_name)

    #For each chamber, we perform a linear regression and store in our analysis.
    #We'll store this temporarily in a dictionary, keyed by each chamber coord (e.g. '1,1')
    linear_regression_analysis_dict = {}
    for chamber_coord in db.get_chamber_coords():
        #get the luminance data for this chamber:
        luminance_data_for_current_chamber = luminance_data[:, np.where(chamber_idxs == chamber_coord)[0][0].item()]

        #perform linear regression:
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(conc_data, luminance_data_for_current_chamber)
        linear_regression_analysis_dict[chamber_coord] = {'slope': slope, 'intercept': intercept, 'r_value': r_value, 'r2':r_value**2, 'p_value': p_value, 'std_err': std_err}

    #store the analysis in the database:
    db.save_new_analysis('standard_0', 'linear_regression', linear_regression_analysis_dict)