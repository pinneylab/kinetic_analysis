from kinetics.functions import *
import numpy as np


def compare_fits(expectation: dict, observed: dict, margin: float = 0.01):
    """ 
    Utility function for determining whether or not two sets of fits are
    equivalent.
    """

    same = True
    for param in expectation.keys():
        e, o = expectation[param], observed[param]
        
        if not o < (1 + margin) * e and o > (1 - margin) * e:
            same = False 
            break
    
    return same


def test_linear_model():
    model = LinearModel() 

    x = np.linspace(0, 100, 50)
    params = {'slope': 1, 'intercept': 1}

    try:
        y = model(x, params)
    except:
        assert False, f'ModelCallError'

    fit = model.fit(x, y)
    assert compare_fits(params, fit)


def test_binding_model():
    model = BindingModel()

    x = np.linspace(0, 100, 50)
    params = {'kd': 1, 'rmax': 1}

    try:
        y = model(x, params)
    except:
        assert False, f'ModelCallError'

    # test fit
    fit = model.fit(x, y)
    assert compare_fits(params, fit)

    fit = model.fit(x, y, fixed_rmax=1)
    assert compare_fits(params, fit)


def test_mm_model():
    model = MichaelisMentenModel()

    x = np.linspace(0, 100, 50)
    params = {'km': 1, 'kcat': 1}

    try:
        y = model(x, params)
    except:
        assert False, f'ModelCallError'

    # test fit
    fit = model.fit(x, y)
    assert compare_fits(params, fit)


def test_single_exponential_model():
    model = SingleExponentialModel()

    x = np.linspace(0, 100, 50)
    params = {'k': -0.05, 'span': 10, 'plateau': 1}

    try:
        y = model(x, params)
    except:
        assert False, f'ModelCallError'

    # test fit
    fit = model.fit(x, y)
    assert compare_fits(params, fit)

