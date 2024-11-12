import numpy as np
import ipywidgets as widgets
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_data(
        xdata: list, 
        ydata: list, 
        xlabel: str, 
        ylabel: str, 
        sharex: bool = False,
        sharey: bool = False,
        n_rows: int = 2, 
        n_cols: int = 2, 
        metadata: list = None, 
        continuous_update: bool = False
        ):
    

    def plot_traces(figure, data: list):
        # clear any previous data
        figure.data = []
        for i, d in enumerate(data):
            figure.layout.annotations[i].update(text=d['title'])

        traces = [go.Scatter(x=d['x'], y=d['y'], mode='markers', showlegend=False) for d in data]

        with figure.batch_update():
            figure.add_traces(
                traces,
                rows=[d['row'] for d in data],
                cols=[d['col'] for d in data]
                )    
            
    n_panels = n_rows * n_cols
    figure = go.FigureWidget(make_subplots(
        rows=n_rows, cols=n_cols, 
        x_title=xlabel, 
        y_title=ylabel,
        subplot_titles=['a'] * n_panels,
        shared_xaxes=sharex,
        shared_yaxes=sharey
        ))
    
    assert len(xdata) == len(ydata)
    if metadata:
        assert len(xdata) == len(metadata)

    # compute grid
    subplot_indices = [(i,j) for i in range(1, n_rows + 1) for j in range(1, n_cols + 1)] * -(-len(xdata) // n_panels)

    # get data for plotting
    data = []
    for x, y, m, (i, j) in zip(xdata, ydata, metadata, subplot_indices):
        data.append({
            'x': x,
            'y': y,
            'row': i,
            'col': j,
            'title': m,
        })

    # batch data
    batched_data = [data[i: i + n_panels] for i in range(0, len(xdata), n_panels)]

    # add traces to figure
    plot_traces(figure, batched_data[0])
    
    # create toggle buttons
    def slider_update_function(update: dict):
        frame = update['new']
        plot_traces(figure, batched_data[frame])
    
    panel_slider = widgets.IntSlider(min=0, max=len(batched_data) - 1, step=1, continuous_update=continuous_update, description='Panel')
    panel_slider.observe(slider_update_function, names='value')

    return widgets.VBox([figure, panel_slider], layout = widgets.Layout(display='flex', align_items='center', justify_content='center', height='500px'))
   

def plot_data_and_fits(
        xdata: list, 
        ydata: list, 
        fits: list,
        model,
        xlabel: str, 
        ylabel: str, 
        sharex: bool = False,
        sharey: bool = False,
        n_rows: int = 2, 
        n_cols: int = 2, 
        metadata: list = None, 
        continuous_update: bool = False,
        ylim: tuple = None
        ):
    

    def plot_traces(figure, data: list):
        # clear any previous data
        figure.data = []
        for i, d in enumerate(data):
            figure.layout.annotations[i].update(text=d['title'])

        traces = [go.Scatter(x=d['x'], y=d['y'], mode='markers', marker=dict(color='black'), showlegend=False) for d in data]
        traces += [go.Scatter(x=d['x_plot'], y=d['y_hat'], mode='lines', showlegend=False) for d in data]

        with figure.batch_update():
            figure.add_traces(
                traces,
                rows=[d['row'] for d in data] * 2,
                cols=[d['col'] for d in data] * 2
                )    
            
    n_panels = n_rows * n_cols
    figure = go.FigureWidget(make_subplots(
        rows=n_rows, cols=n_cols, 
        x_title=xlabel, 
        y_title=ylabel,
        subplot_titles=['a'] * n_panels,
        shared_yaxes=sharey,
        shared_xaxes=sharex,
        ))
    
    if ylim:
        figure.update_yaxes(range=[ylim[0], ylim[1]])
        
    # compute grid
    subplot_indices = [(i,j) for i in range(1, n_rows + 1) for j in range(1, n_cols + 1)] * -(-len(xdata) // n_panels)

    # get data for plotting
    data = []
    for x, y, params, m, (i, j) in zip(xdata, ydata, fits, metadata, subplot_indices):
        x_plot = np.linspace(min(x), max(x), 1000)
        data.append({
            'x': x,
            'y': y,
            'x_plot': x_plot,
            'y_hat': model(x_plot, params),
            'row': i,
            'col': j,
            'title': m,
        })

    # batch data
    batched_data = [data[i: i + n_panels] for i in range(0, len(xdata), n_panels)]

    # add traces to figure
    plot_traces(figure, batched_data[0])
    
    # create toggle buttons
    def slider_update_function(update: dict):
        frame = update['new']
        plot_traces(figure, batched_data[frame])
    
    panel_slider = widgets.IntSlider(min=0, max=len(batched_data) - 1, step=1, continuous_update=continuous_update, description='Panel')
    panel_slider.observe(slider_update_function, names='value')

    return widgets.VBox([figure, panel_slider], layout = widgets.Layout(display='flex', align_items='center', justify_content='center', height='500px'))



