"""Grid creator.

Grid creator so making grids is somewhat easier than writing it as a raw numpy
array. Credit to Tom v. Meer for writing this up.
"""
import ast

from flask import Flask, render_template, request
from flask_socketio import SocketIO

# World may not be importable, depending on how you have set up your
# conda/pip/venv environment. Here we try to fix that by forcing the world to
# be in your python path. If it still doesn't work, come to a tutorial, look up
# how to fix module import errors, or ask ChatGPT.
try:
    from world import Grid
    from world import GRID_CONFIGS_FP
except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys

    root_path = path.abspath(path.join(
        path.join(path.abspath(__file__), pardir), pardir)
    )

    if root_path not in sys.path:
        sys.path.append(root_path)

    from world import Grid
    from world import GRID_CONFIGS_FP


# Initialize SocketIO App
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socket_io = SocketIO(app)


def draw_grid(grid):
    """Creates a JSON payload which will be displayed in the browser."""
    materials = {0: 'cell_empty',
                 1: 'cell_boundary',
                 2: 'cell_obstacle',
                 3: 'cell_target',
                 4: 'cell_charger',
                 5: 'cell_kitchen',
                 6: 'cell_table',}
    
    return {'grid': render_template(
        'grid.html',
        height=30, width=30,
        n_rows=grid.n_rows, n_cols=grid.n_cols,
        room_config=grid.cells, materials=materials
    )}


# Routes
@app.route('/')
def home():
    return render_template("editor.html")


@app.route('/build_grid')
def build_grid():
    """Main route for building a grid.

    Given a request with the following parameters, a grid and accompanying
    statistics are being constructed.

    Request params:
        height: number of rows in the grid.
        width: number of columns in the grid.
        obstacles: a list of tuples (x,y) of obstacle locations.
        targets: a list of tuples (x,y) of target locations.
        chargers: a list of tuples (x,y) of charger locations.
        walls: a list of tuples (x,y) of wall locations.
        kitchens: a list of tuples (x,y) of kitchen locations.
        tables: a list of tuples (x,y) of table locations.
        save: boolean (true, false) to save the current grid to a file.
        name: filename to save the current grid to.
     """
    
    # Get properties of the grid from the request
    n_rows = int(request.args.get('height'))
    n_cols = int(request.args.get('width'))
    obstacles = ast.literal_eval(request.args.get('obstacles', '[]'))
    targets = ast.literal_eval(request.args.get('targets', '[]'))
    chargers = ast.literal_eval(request.args.get('chargers', '[]'))
    kitchens = ast.literal_eval(request.args.get('kitchens', '[]'))
    tables = ast.literal_eval(request.args.get('tables', '[]'))
    to_save = False if request.args.get('save') == 'false' else True
    name = str(request.args.get('name'))

    # Put all of those things on to a Grid object
    grid = Grid(n_cols, n_rows)
    for (x, y) in obstacles:
        grid.place_object(x, y, "obstacle")
    for (x, y) in targets:
        grid.place_object(x, y, "target")
    for (x, y) in chargers:
        grid.place_object(x, y, "charger")
    for (x, y) in tables:
        grid.place_object(x, y, "tables")
    for (x, y) in kitchens:
        grid.place_object(x, y, "kitchens")

    drawn_grid = draw_grid(grid)

    # If we need to save it, do so
    if to_save and len(name) > 0:
        save_fp = GRID_CONFIGS_FP / f"{name}.npy"
        grid.save_grid_file(save_fp)
        drawn_grid["success"] = "true"
        drawn_grid["save_fp"] = str(save_fp)

    return drawn_grid


if __name__ == '__main__':
    socket_io.run(app, debug=False, allow_unsafe_werkzeug=True)