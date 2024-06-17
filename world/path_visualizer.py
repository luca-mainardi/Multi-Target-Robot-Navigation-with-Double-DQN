"""Path Visualizer.

This script is used to visualize the path of the agents in the environment.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""

from colorcet import bmw, glasbey_hv
from PIL import Image
from PIL import ImageDraw
import numpy as np

# World may not be importable, depending on how you have set up your
# conda/pip/venv environment. Here we try to fix that by forcing the world to
# be in your python path. If it still doesn't work, come to a tutorial, look up
# how to fix module import errors, or ask ChatGPT.
try:
    from world import GUI
except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys

    root_path = path.abspath(
        path.join(path.join(path.abspath(__file__), pardir), pardir)
    )

    if root_path not in sys.path:
        sys.path.append(root_path)

    from world import GUI


def draw_base_image(
    cells: np.ndarray,
    scalar: int,
    image_size: tuple[int, int],
) -> Image.Image:
    """Draws the base image containing the grid and objects on the grid.

    Args:
        cells: The cell array underlying the grid representation of the
            environment.
        scalar: How much to scale the original grid by in the output image.
        image_size: Output image size.

    Returns:
        An RGBA image with the grid on it.
    """
    grid_size = cells.shape
    base_image = Image.new(mode="RGBA", size=image_size, color=(255, 255, 255, 255))
    draw = ImageDraw.ImageDraw(base_image)
    for row in range(grid_size[1]):
        y = (row * scalar) + 1
        for col in range(grid_size[0]):
            x = (col * scalar) + 1
            val = cells[col, row]
            color = GUI.CELL_COLORS[val]

            draw.rectangle(
                (x, y, x + scalar, y + scalar), color, outline=(255, 255, 255)
            )

    return base_image


def draw_starting_square(
    starting_square: tuple[int, int], grid_scalar: int, image_size: tuple[int, int]
):
    """Draws the starting square as a yellow square."""
    square_image = Image.new(mode="RGBA", size=image_size, color=(255, 255, 255, 0))
    draw = ImageDraw.ImageDraw(square_image)
    x = (starting_square[0] * grid_scalar) + 1
    y = (starting_square[1] * grid_scalar) + 1

    draw.rectangle(
        (x, y, x + grid_scalar, y + grid_scalar),
        (242, 211, 82),
        outline=(255, 255, 255),
    )

    return square_image


def draw_freq_image(
    agent_path: list[tuple[int, int]],
    grid_shape: tuple[int, int],
    grid_scalar: int,
    freq_scalar: int,
    image_size: tuple[int, int],
) -> Image.Image:
    """Draws the cell visit frequency image.

    Args:
        agent_path: The path that each agent took through the environment.
        grid_shape: The actual shape of the grid.
        grid_scalar: The size of each grid cell to draw. For example, a value of
            30 would result in an image where each grid cell is 30x30 px.
        freq_scalar: The size of each frequency color square to draw. For
            example, in a 30x30 grid square, if we set the freq_scalar to 20,
            then the freq color square we draw is 20x20 centered in the middle
            of the grid square.
        image_size: The size of the final image to draw.

    Returns:
        An image that is transparent except where the frequency squares are.
    """
    # Create the frequency grid array to figure out how often a cell if
    # traversed
    freq_grid = np.zeros(grid_shape, dtype=float)
    for pos in agent_path:
        freq_grid[pos] += 1.0

    # Normalize by the max value to 0-255
    freq_grid /= np.max(freq_grid)
    freq_grid *= 255.0
    freq_grid = freq_grid.astype(int)

    cell_offset = (grid_scalar - freq_scalar) // 2

    freq_image = Image.new(mode="RGBA", size=image_size, color=(255, 255, 255, 0))
    draw = ImageDraw.ImageDraw(freq_image)

    for row in range(grid_shape[1]):
        y = (row * grid_scalar) + 1 + cell_offset
        for col in range(grid_shape[0]):
            x = (col * grid_scalar) + 1 + cell_offset
            val = freq_grid[col, row]
            if val == 0:
                # Don't draw anything if the cells has never been traversed.
                continue
            try:
                # minus because we want to start from white.
                color = bmw[-val]
            except IndexError:
                # There is no chance the value is < 1 here, but just in case.
                color = bmw[0]

            draw.rectangle((x, y, x + freq_scalar, y + freq_scalar), color)
    return freq_image


def draw_path(
    agent_path: list[tuple[int, int]],
    grid_scalar: int,
    base_line_width: int,
    line_color: tuple[int, int, int],
    image_size: tuple[int, int],
    visit_frequencies: dict[tuple[int, int], int],
) -> Image.Image:
    """Draws the path of each agent on the grid, adjusting the line width based on visit frequency.

    Args:
        agent_path: The path that the agent took through the environment.
        grid_scalar: The size of each grid cell to draw. For example, a value of
            30 would result in an image where each grid cell is 30x30 px.
        base_line_width: The base width of the path line to draw.
        line_color: The color of the path line to draw.
        image_size: The size of the final image to draw.
        visit_frequencies: A dictionary with the frequency of visits to each cell.

    Returns:
         An image that is transparent except where the line paths are.
    """

    path_image = Image.new(mode="RGBA", size=image_size, color=(255, 255, 255, 0))
    draw = ImageDraw.ImageDraw(path_image)

    for i in range(len(agent_path) - 1):
        start = agent_path[i]
        end = agent_path[i + 1]
        offset = grid_scalar // 2
        start_x = (start[0] * grid_scalar) + offset
        start_y = (start[1] * grid_scalar) + offset
        end_x = (end[0] * grid_scalar) + offset
        end_y = (end[1] * grid_scalar) + offset

        # Determine line width based on visit frequency
        freq = visit_frequencies.get(start, 1)
        line_width = base_line_width * freq

        draw.line((start_x, start_y, end_x, end_y), fill=line_color, width=line_width)

    return path_image


def float_rgb_to_int(rgb: tuple[float, float, float]) -> tuple[int, int, int]:
    """Converts an RGB tuple of floats to an RGB tuple of ints.

    Args:
        rgb: The RGB tuple of floats.

    Returns:
        The RGB tuple of ints.
    """
    return tuple(int(c * 255) for c in rgb)


def visualize_path(
    grid_cells: np.ndarray, agent_path: list[list[tuple[int, int]]]
) -> list[Image.Image]:
    """Visualizes the path of (multiple) agents through the environment.

    Args:
        grid_cells: The grid cells that are underlying the Grid object.
        agent_paths: A list of tuples containing the x and y coordinates of
            the agent's path.

    Returns:
        A list of images showing the grid and the frequency of the agent
        traversing each position on the grid.
    """
    grid_size = grid_cells.shape
    scalar = 30
    image_size = tuple((g * scalar) + 2 for g in grid_size)
    freq_scalar = 20

    base_image = draw_base_image(grid_cells, scalar, image_size)

    starting_square_img = draw_starting_square(agent_path[0], scalar, image_size)

    # Create visit frequencies map
    visit_frequencies = {}
    for pos in agent_path:
        if pos in visit_frequencies:
            visit_frequencies[pos] += 1
        else:
            visit_frequencies[pos] = 1

    # Generate path image with varying line widths based on visit frequency
    path_image = draw_path(
        agent_path,
        scalar,
        1,
        float_rgb_to_int(glasbey_hv[0]),
        image_size,
        visit_frequencies,
    )

    img = Image.alpha_composite(base_image, starting_square_img)
    img = Image.alpha_composite(img, path_image)

    # img = Image.alpha_composite(base_image, starting_square_img)
    # freq_image = draw_freq_image(agent_path, grid_size, scalar, freq_scalar,
    # image_size)

    return img
