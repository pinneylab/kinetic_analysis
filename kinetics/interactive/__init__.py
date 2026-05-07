from flask import Flask, render_template, request, jsonify
import logging
import numpy as np
import os

base_dir = os.path.abspath(os.path.dirname(__file__))


def _batch_data(xdata, ydata, zdata, inclusion_masks, xplot, yplot, zplot, titles, n_rows, n_cols):
    
    assert len(xdata) == len(ydata)

    surface = False
    if len(zdata) > 0:
        assert len(zdata) == len(xdata)
        surface = True
    else:
        zdata = np.full((len(xdata), len(xdata[0])), None)

    fit_traces = False
    if len(xplot) > 0:
        assert len(xplot) == len(yplot)
        fit_traces = True

        if surface:
            assert len(xplot) == len(zplot)
        else:
            zplot = [None] * len(xplot)
    else:
        xplot = [None] * len(xdata)
        yplot = [None] * len(xdata)
        zplot = [None] * len(xdata)
    
    if len(titles) > 0:
        assert len(titles) == len(xdata)
    else:
        titles = list(range(len(xdata)))

    if len(inclusion_masks) > 0:
        assert len(inclusion_masks) == len(xdata)
    else:
        inclusion_masks = [[True] * len(x) for x in xdata]

    # compute grid
    n_panels = n_cols * n_rows
        # compute grid
    subplot_indices = [(i,j) for i in range(0, n_rows) for j in range(0, n_cols)] * -(-len(xdata) // n_panels)

    # get data for plotting
    print(len(xdata), len(ydata), len(zdata), len(inclusion_masks), len(xplot), len(yplot), len(zplot), len(titles))
    data = []
    for x, y, z, m, xp, yp, zp, t, (i, j) in zip(xdata, ydata, zdata, inclusion_masks, xplot, yplot, zplot, titles, subplot_indices):
        d = {}
        d['i'], d['j'] = i, j
        d['layout'] = {'title': {'text': t, 'font': {'size': 14}}, 'margin': {'l': 40, 'r': 40, 't': 40, 'b': 40}, 'showlegend': False}

        d['plotting_data'] = [] 
        d['plotting_data'].append({
            'x': x.tolist(),
            'y': y.tolist(),
            'z': z.tolist(),
            'type': 'scatter' if not surface else 'scatter3d', 
            'mode': 'markers',
            'inclusion_mask': m
        })

        if fit_traces and not surface:
            d['plotting_data'].append({
                'x': xp.tolist(),
                'y': yp.tolist(),
                'mode': 'lines',
                'type': 'scatter',
                'line': {'color': 'black'}
            })
        elif fit_traces and surface:
            d['plotting_data'].append({
                'x': xp.tolist(),
                'y': yp.tolist(),
                'z': zp.tolist(),
                'type': 'surface'
            })

            assert xp.shape == yp.shape == zp.shape

        data.append(d)
    # batch data
    batched_data = [data[i: i + n_panels] for i in range(0, len(xdata), n_panels)]

    return batched_data


# def _init_app(xdata, ydata, zdata, inclusion_masks, model, fits, titles, n_rows, n_cols, xlabel, ylabel):

#     batched_data = _batch_data(xdata, ydata, zdata, inclusion_masks, model, fits, titles, n_rows, n_cols)
#     app = Flask(__name__, template_folder=os.path.join(base_dir, 'templates'), static_folder=os.path.join(base_dir, 'static'))

#     # Suppress Werkzeug logs
#     # log = logging.getLogger('werkzeug')
#     # log.setLevel(logging.ERROR)  # Suppress all logs below ERROR level

#     @app.route('/')
#     def index():
#         """Render the main page with navigation."""
#         return render_template('index.html', rows=n_rows, cols=n_cols, panels=len(batched_data), xlabel=xlabel, ylabel=ylabel)
    
#     @app.route('/data/<int:index>')
#     def plot(index):

#         if index < 0 or index >= len(batched_data):
#             return "Invalid panel index", 400
        
#         batch = batched_data[index]
#         return jsonify(batch)

#     return app


# def launch_interactive_plot(
#         xdata, 
#         ydata, 
#         zdata: list = [],
#         model = None, 
#         fits: list = [],
#         titles: list = [], 
#         n_rows: int = 2, 
#         n_cols: int = 3, 
#         xlabel: str = 'x', 
#         ylabel: str = 'y',
#         ):
    
#     # TODO: allow for masking of individual points
#     inclusion_masks = []

#     # TODO: enable user to toggle between linear and log axis scaling
#     xscale = 'linear'
#     yscale = 'linear'
#     zscale = 'linear'

#     # TODO: enable user to set the range of x and y axes
#     xrange = (None, None)
#     yrange = (None, None)
#     zrange = (None, None)

#     app = _init_app(xdata, ydata, zdata, inclusion_masks, model, fits, titles, n_rows, n_cols, xlabel, ylabel)
#     app.run(debug=False, use_reloader=False)

def _init_app(batched_data, n_rows, n_cols, xlabel, ylabel):

    # batched_data = _batch_data(xdata, ydata, zdata, inclusion_masks, xplot, yplot, zplot, titles, n_rows, n_cols)
    app = Flask(__name__, template_folder=os.path.join(base_dir, 'templates'), static_folder=os.path.join(base_dir, 'static'))

    # Suppress Werkzeug logs
    # log = logging.getLogger('werkzeug')
    # log.setLevel(logging.ERROR)  # Suppress all logs below ERROR level

    @app.route('/')
    def index():
        """Render the main page with navigation."""
        return render_template('index.html', rows=n_rows, cols=n_cols, panels=len(batched_data), xlabel=xlabel, ylabel=ylabel)
    
    @app.route('/data/<int:index>')
    def plot(index):

        if index < 0 or index >= len(batched_data):
            return "Invalid panel index", 400
        
        batch = batched_data[index]
        return jsonify(batch)

    return app


def launch_interactive_plot(
        xdata, 
        ydata, 
        zdata: list = [],
        xplot: list = [],
        yplot: list = [],
        zplot: list = [],
        titles: list = [], 
        n_rows: int = 2, 
        n_cols: int = 3, 
        xlabel: str = 'x', 
        ylabel: str = 'y',
        ):
    
    # TODO: allow for masking of individual points
    inclusion_masks = []

    # TODO: enable user to toggle between linear and log axis scaling
    xscale = 'linear'
    yscale = 'linear'
    zscale = 'linear'

    # TODO: enable user to set the range of x and y axes
    xrange = (None, None)
    yrange = (None, None)
    zrange = (None, None)

    app = _init_app(xdata, ydata, zdata, inclusion_masks, xplot, yplot, zplot, titles, n_rows, n_cols, xlabel, ylabel)
    app.run(debug=False, use_reloader=False)


def _init_model_simulator_app(xarray, model, params, yarray, xlabel, ylabel, zlabel, surface_plot, slider_config):

    # render HTML template
    # sliders
    app = Flask(
        __name__, 
        template_folder=os.path.join(base_dir, 'templates'), 
        static_folder=os.path.join(base_dir, 'static'),
        )
    
    # Suppress Werkzeug logs
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)  # Suppress all logs below ERROR level
    

    @app.route('/send_data', methods=['GET'])
    def send_data():
        response_data = {
            "params": params,
            "xarray": xarray.tolist(),
            "yarray": yarray.tolist(),
            "slider_config": slider_config
        }
        return jsonify(response_data)  # Send the data as JSON


    @app.route('/')
    def index():
        """Render the main page with navigation."""
        return render_template(
            'model_simulator.html', 
            xarray=xarray.tolist(),
            params=params,
            yarray=yarray.tolist(),
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            surface_plot=surface_plot
            )


    @app.route('/simulate', methods=['POST'])
    def simulate():
        params = request.get_json()
        prediction = model(xarray, yarray, params) if surface_plot else model(xarray, params)
        return jsonify({'prediction': prediction.tolist()}), 200
    
    return app
    

def launch_model_simulator(
    xarray: np.ndarray,
    model,
    params: dict,
    yarray: np.ndarray = np.array([]),
    xlabel: str = 'x',
    ylabel: str = 'y',
    zlabel: str = 'z',
    slider_config: dict = {},
):
    
    assert isinstance(xarray, np.ndarray) and isinstance(yarray, np.ndarray), 'X and Y arrays must be numpy arrays.'
    
    surface_plot = False
    if yarray.shape != (0, ):
        assert xarray.shape == yarray.shape, 'X and Y arrays must be the same size.'
        surface_plot = True

    for param_name, param_value in params.items():
        if param_name in slider_config.keys():
            continue
        slider_config[param_name] = {'scale': 'log', 'min': 0, 'max': 1000, 'stepsize': 1}

    app = _init_model_simulator_app(xarray, model, params, yarray, xlabel, ylabel, zlabel, surface_plot, slider_config)
    app.run(debug=False, use_reloader=False)