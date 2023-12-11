import plotly.graph_objs as go
import base64
import tempfile
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from dash import Dash, dcc, html, Input, Output, no_update
#from dash import jupyter_dash
#jupyter_dash.default_mode="external"


def plot(db, run_name:str, analysis_name:str, plotting_var:str, title:str=None):
    ''' This function creates a Dash visualization of a chip, based on a certain Run (run_name)
        Inputs:
            db: a Database object
            run_name: the name of the run to analyze
            analysis_name: the name of the analysis to perform
            plotting_var: the variable to be plotted for each chamber
            title: a string to be used as the title of the plot

    '''

    #jupyter_dash.infer_jupyter_proxy_config()

    #get mapping of coord to chamber name:
    chamber_coords = db.get_chamber_coords()
    chamber_names = db.get_chamber_names()
    chamber_names_dict = {chamber_coords[i]: chamber_names[i] for i in range(len(chamber_coords))}

    #get the analysis:
    analysis_chambers = db.get(Path('runs')/run_name/'analyses'/analysis_name/'chambers')

    #convert serialized values to Quantity objects:
    analysis_chambers = db._make_quantities_from_serialized_dict(analysis_chambers)

    #get the plotting values:
    plotting_values = {}
    for chamber_coord, chamber_name in chamber_names_dict.items():
        plotting_values[chamber_coord] = analysis_chambers[chamber_coord][plotting_var]
    
    #plot the chip:chamber_names_dict
    if title is not None:
        title = title + ': ' + plotting_var
    _plot_chip(plotting_values, chamber_names_dict, title=title)

def _plot_chip(plotting_values, chamber_names, graphing_function=None, title=None):
    ''' This function creates a Dash visualization of a chip, based on a certain Run (run_name)
        Inputs:
            plotting_values: a dictionary mapping chamber_coord to the variable to be plotted for that chamber
            chamber_names: a dictionary mapping chamber_coord to the name of the sample in the chamber (e.g. '1,1': ecADK_XYZ')
            graphing_function: a function that takes in a single chamber_id (e.g. '1,1') and matplotlib axis and returns the axis object after plotting.
            title: a string to be used as the title of the plot
        TODO: make all the variables stored in Dash properly...
    '''
    # Make the image array
    #NB: eventually, store width/height in DB and reference!
    img_array = np.zeros([56,32])

    for chamber_id, value in plotting_values.items():
        #make sure it's a single value, not an array:
        magnitude = value.magnitude
        x = int(chamber_id.split(',')[0])
        y = int(chamber_id.split(',')[1])
        img_array[y-1,x-1] = magnitude 
    #add back the units:
    img_array = img_array * value.units
    
    #generate title
    if title is None:
        title = ''
    
    #Create the figure
    layout = go.Layout()
    fig = go.Figure(layout=layout, data=go.Heatmap(z=img_array, colorscale='Viridis',
                                                   colorbar = dict(title=str(value.units))))
    #center title in fig
    fig.update_layout(title=title,
                        title_x=0.5, 
                        yaxis=dict(scaleanchor="x", scaleratio=1, autorange='reversed'), 
                        xaxis=dict(scaleratio=1),
                        plot_bgcolor='rgba(0,0,0,0)',
                        width=600, height=600,
                        hovermode='x'
                        )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    #create dash app:
    app = Dash(__name__)
    app.layout = html.Div([
        dcc.Graph(id="graph", figure=fig, clear_on_unhover=True),
        dcc.Tooltip(id="graph-tooltip"),
    ])

    ### GRAPHING FUNCTION ON HOVER:
    if graphing_function is not None:
        @app.callback(
            Output("graph-tooltip", "show"),
            Output("graph-tooltip", "bbox"),
            Output("graph-tooltip", "children"),
            Input("graph", "hoverData"),
        )
        def display_hover(hoverData):
            if hoverData is None:
                return False, no_update, no_update
            print(hoverData)
            # demo only shows the first point, but other points may also be available
            pt = hoverData["points"][0]
            chamber_id = str(pt['x']+1) + ',' + str(pt['y']+1)
            bbox = pt["bbox"]
            chamber_name = chamber_names[chamber_id]
            #get the data for the point:
            fig, ax = plt.subplots()
            ax = graphing_function(chamber_id, ax)
            #reduce whitespace on margins of graph:
            fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0, hspace=0)
            #save the figure as a temp file:
            tempfile_name = tempfile.NamedTemporaryFile().name+'.png'
            plt.savefig(tempfile_name)
            plt.close()
            # #read in temp file as base64 encoded string:
            with open(tempfile_name, "rb") as image_file:
                img_src = "data:image/png;base64," + str(base64.b64encode(image_file.read()).decode("utf-8"))
            children = [
                html.Div(children=[
                    #no space after header:
                    html.H3('{},{}:  {}'.format(pt['x'], pt['y'], chamber_name), style={"color": 'black', "fontFamily":"Arial", "textAlign": "center", "marginBottom": "0px"}),
                    #add the image with reduced whitespace:
                    html.Img(src=img_src, style={"width": "100%"}),
                ],
                style={'width': '400px', 'white-space': 'none'})
            ]
            return True, bbox, children
    #app.run_server(jupyter_mode="inline", debug=False)
    #return app
    app.run_server()