from flask import Flask, render_template, request, jsonify
import numpy as np
import os

base_dir = os.path.abspath(os.path.dirname(__file__))

def _batch_data(xdata, ydata, inclusion_masks, model, fits, titles, n_rows, n_cols):
    
    assert len(xdata) == len(ydata)

    fit_traces = False
    if len(fits) > 0:
        assert len(fits) == len(xdata)
        assert model
        fit_traces = True
    else:
        fits = [None] * len(xdata)
    
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
    data = []
    for x, y, m, f, t, (i, j) in zip(xdata, ydata, inclusion_masks, fits, titles, subplot_indices):
        d = {}
        d['i'], d['j'] = i, j
        d['layout'] = {'title': {'text': t, 'font': {'size': 14}}, 'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}, 'showlegend': False}

        d['plotting_data'] = [] 
        d['plotting_data'].append({
            'x': x.tolist(),
            'y': y.tolist(),
            'mode': 'markers',
            'inclusion_mask': m
        })

        if fit_traces:
            x_plot = np.linspace(min(x), max(x), 1000)
            d['plotting_data'].append({
                'x': x_plot.tolist(),
                'y': model(x_plot, f).tolist(),
                'mode': 'lines',
                'type': 'scatter',
                'line': {'color': 'black'}
            })

        data.append(d)

    # batch data
    batched_data = [data[i: i + n_panels] for i in range(0, len(xdata), n_panels)]

    return batched_data


def _init_app(xdata, ydata, inclusion_masks, model, fits, titles, n_rows, n_cols, xlabel, ylabel):

    batched_data = _batch_data(xdata, ydata, inclusion_masks, model, fits, titles, n_rows, n_cols)
    app = Flask(__name__, template_folder=os.path.join(base_dir, 'templates'), static_folder=os.path.join(base_dir, 'static'))

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


def plot_data(
        xdata, 
        ydata, 
        model = None, 
        fits: list = [],
        titles: list = [], 
        n_rows: int = 2, 
        n_cols: int = 3, 
        xlabel: str = 'x', 
        ylabel: str = 'y'
        ):
    
    # TODO: allow for masking of individual points
    inclusion_masks = []

    app = _init_app(xdata, ydata, inclusion_masks, model, fits, titles, n_rows, n_cols, xlabel, ylabel)
    app.run(debug=False, use_reloader=False)