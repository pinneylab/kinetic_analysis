from kinetics.functions import *
from kinetics.interactive import *
from flask import Flask, render_template, request, jsonify

def batch_data(xdata, ydata, titles, n_rows, n_cols, sharex, sharey):

    # compute grid
    n_panels = n_cols * n_rows
        # compute grid
    subplot_indices = [(i,j) for i in range(0, n_rows) for j in range(0, n_cols)] * -(-len(xdata) // n_panels)

    # get data for plotting
    data = []
    for x, y, t, (i, j) in zip(xdata, ydata, titles, subplot_indices):
        d = {}
        d['i'], d['j'] = i, j
        d['layout'] = {'title': {'text': t, 'font': {'size': 14}}, 'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}}

        d['plotting_data'] = {
            'x': x.tolist(),
            'y': y.tolist(),
            'mode': 'markers'
        }
        data.append(d)

    # batch data
    batched_data = [data[i: i + n_panels] for i in range(0, len(xdata), n_panels)]

    return batched_data

n = 100

model = MichaelisMentenModel()
params = [{'kcat': kcat, 'km': 1} for kcat in np.random.normal(100, 50, size=n)]

xdata = [np.linspace(0, 10, 20)] * n
ydata = [model(x, p) for x,p in zip(xdata, params)]

titles = ['{} / {:.2f}'.format(i, f['kcat']) for i, f in enumerate(params)]

xlabel = '''HELLO'''
ylabel = '''WORLD'''
n_rows, n_cols = 2, 3
batched_data = batch_data(xdata, ydata, titles, n_rows, n_cols, False, False)
print(len(batched_data))

app = Flask(__name__)

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

app.run(debug=True, use_reloader=False)