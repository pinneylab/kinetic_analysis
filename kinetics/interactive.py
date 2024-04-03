import ipywidgets as widgets
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_data(figure, data: list):

    # clear any previous data
    figure.data = []
    for i, d in enumerate(data):
        figure.layout.annotations[i].update(text=d['title'])

    traces = [go.Scatter(x=d['x'], y=d['y'], mode='markers', marker=dict(color='black'), showlegend=False) for d in data]
    traces += [go.Scatter(x=d['x'], y=d['y_hat'], mode='lines', showlegend=False) for d in data]

    with figure.batch_update():
        figure.add_traces(
            traces,
            rows=[d['row'] for d in data] * 2,
            cols=[d['col'] for d in data] * 2
            )
    

def make_figure(models: list, xlabel: str, ylabel: str, n_rows: int = 2, n_cols: int = 2, metadata: list = None, continuous_update: bool = False):
    
    n_panels = n_rows * n_cols
    figure = go.FigureWidget(make_subplots(
        rows=n_rows, cols=n_cols, 
        x_title=xlabel, 
        y_title=ylabel,
        subplot_titles=['a'] * n_panels
        ))

    # compute grid
    subplot_indices = [(i,j) for i in range(1, n_rows + 1) for j in range(1, n_cols + 1)] * -(-len(models) // n_panels)

    # get data for plotting
    data = []
    for model, m, (i, j) in zip(models, metadata, subplot_indices):
        data.append({
            'x': model.x,
            'y': model.y,
            'y_hat': model(model.x),
            'row': i,
            'col': j,
            'title': m,
        })

    # group models into sublists length n_rows * n_cols
    batched_data = [data[i: i + n_panels] for i in range(0, len(models), n_panels)]

    # add traces to figure
    plot_data(figure, batched_data[0])
    
    # create toggle buttons
    def slider_update_function(update: dict):
        frame = update['new']
        plot_data(figure, batched_data[frame])
    
    panel_slider = widgets.IntSlider(min=0, max=len(batched_data) - 1, step=1, continuous_update=continuous_update, description='Panel')
    panel_slider.observe(slider_update_function, names='value')

    return widgets.VBox([figure, panel_slider], layout = widgets.Layout(display='flex', align_items='center', justify_content='center', height='500px'))
    