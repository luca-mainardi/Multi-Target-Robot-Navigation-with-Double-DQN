"""GUI.

Provides a GUI for the environment using pygame.
"""
import sys
from time import perf_counter, sleep

import numpy as np
import pygame
from pygame import gfxdraw


class GUI:
    CELL_COLORS = [
        (255, 255, 255),  # Empty cell
        (57, 57, 57),     # Boundary cell
        (57, 57, 57),     # Obstacle cell
        (34, 139, 34),    # Target cell
        (255, 119, 0),    # Charger cell
        (255, 255, 0),    # Kitchen cell
        (139, 69, 19),    # Table cell
    ]
    INFO_NAME_MAP = [
        ("cumulative_reward", "Cumulative reward:"),
        ("total_steps", "Total steps:"),
        ("total_failed_move", "Total failed moves:"),
        ("agent_storage_level", "Agent Storage Level:"),

        # ("total_targets_reached", "Total targets reached:"),
        ("fps", "FPS:"),
    ]

    def __init__(self, grid_size: tuple[int, int],
                 window_size: tuple[int, int] = (1152, 768)):
        """Provides a GUI to show what is happening in the environment.

        Args:
            grid_size: (n_cols, n_rows) in the grid.
            window_size: The size of the pygame window. (width, height).
        """
        self.grid_size = grid_size

        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Data Intelligence Challenge 2024")
        self.clock = pygame.time.Clock()

        self.stats = self._reset_stats()

        self.grid_panel_size = (int(window_size[0] * 0.75), window_size[1])
        self.info_panel_rect = pygame.Rect(
            self.grid_panel_size[0],
            0,
            window_size[0] - self.grid_panel_size[0],
            window_size[1]
        )
        self.last_agent_pos = None

        self.paused = False
        self.paused_clicked = False
        self.step = False
        self.step_clicked = False

        # Find the smallest window dimension and max grid size to calculate the
        # grid scalar
        self.scalar = min(self.grid_panel_size)
        self.scalar /= max(self.grid_size) * 1.2

        # FPS timer
        self.last_render_time = perf_counter()
        self.last_10_fps = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.frame_count = 0

        self._initial_render()

    def reset(self):
        """Called during the reset method of the environment."""
        self.stats = self._reset_stats()
        self._initial_render()

    @staticmethod
    def _reset_stats():
        return {"total_targets_reached": 0,
                "total_failed_move": 0,
                "fps": "0.0",
                "total_steps": 0,
                "cumulative_reward": 0,
                "agent_storage_level": 0}

    def _initial_render(self):
        """Initial render of the environment. Also shows loading text."""
        background = pygame.Surface(self.window.get_size())
        background = background.convert()
        background.fill((250, 250, 250))

        # Display the loading text
        font = pygame.font.Font(None, 36)
        text = font.render("Loading environment...", True, (10, 10, 10))
        textpos = text.get_rect()
        textpos.centerx = background.get_rect().centerx
        textpos.centery = background.get_rect().centery

        # Blit the text onto the background surface
        background.blit(text, textpos)
        # Blit the background onto the window
        update_rect = self.window.blit(background, background.get_rect())
        # Tell pygame to update the display where the window was blit-ed
        pygame.display.update(update_rect)

    @staticmethod
    def _downsample_rect(rect: pygame.Rect, scalar: float) -> pygame.Rect:
        """Downsamples the given rectangle by a scalar."""
        x = rect.x * scalar
        y = rect.y * scalar
        width = rect.width * scalar
        height = rect.height * scalar
        return pygame.Rect(x, y, width, height)

    def _draw_grid(self, surface: pygame.Surface, grid: np.ndarray,
                   x_offset: int, y_offset: int):
        """Draws the grid world on the given surface."""
        for row in range(grid.shape[1]):
            y = (row * self.scalar) + y_offset
            for col in range(grid.shape[0]):
                x = (col * self.scalar) + x_offset
                val = grid[col, row]

                rect = pygame.Rect(x, y, self.scalar, self.scalar)
                pygame.draw.rect(surface, self.CELL_COLORS[val], rect)
                pygame.draw.rect(surface, (255, 255, 255), rect, width=1)

    def _draw_button(self, surface: pygame.Surface, text: str,
                     rect: pygame.Rect, color: tuple[int, int, int],
                     text_color: tuple[int, int, int] = (0, 0, 0)):
        """Draws a button on the given surface."""
        pygame.draw.rect(surface, color, rect)
        pygame.draw.rect(surface, (255, 255, 255), rect, width=1)

        font = pygame.font.Font(None, int(self.scalar / 2))
        text = font.render(text, True, text_color)
        textpos = text.get_rect()
        textpos.centerx = rect.centerx
        textpos.centery = rect.centery
        surface.blit(text, textpos)

    def _draw_agent(self, surface: pygame.Surface,
                    agent_pos: list[tuple[int, int]],
                    x_offset: int, y_offset: int):
        """Draws the agent on the grid world."""

        # Draw the agent as a gray circle
        x = (agent_pos[0] * self.scalar) + x_offset
        y = (agent_pos[1] * self.scalar) + y_offset
        r = int(self.scalar / 2) - 8
        rect = pygame.Rect(x + 4, y + 4, self.scalar - 8, self.scalar - 8)
        gfxdraw.aacircle(surface, rect.centerx, rect.centery, r,
                            (0, 0, 102))
        gfxdraw.filled_circle(surface, rect.centerx, rect.centery, r,
                                (0, 0, 102))

    def _draw_info(self, surface) -> tuple[pygame.Rect, pygame.Rect]:
        """Draws the info panel on the surface.

        Returns:
            The rect of the pause button and the step button.
        """
        x_offset = self.grid_panel_size[0] + 20
        y_offset = 50

        col_width = 200
        row_height = 30

        font = pygame.font.Font(None, 24)
        for row, (key, name) in enumerate(self.INFO_NAME_MAP):
            y_pos = y_offset + (row * row_height)
            text = font.render(name, True, (0, 0, 0))
            textpos = text.get_rect()
            textpos.x = x_offset
            textpos.y = y_pos
            surface.blit(text, textpos)

            text = font.render(f"{self.stats[key]}", True, (0, 0, 0))
            textpos = text.get_rect()
            textpos.x = x_offset + col_width
            textpos.y = y_pos
            surface.blit(text, textpos)

        button_row = len(self.INFO_NAME_MAP) + 1
        # Draw a button to pause the simulation
        pause_rect = pygame.Rect(x_offset,
                                 y_offset + (button_row * row_height) + 50,
                                 200,
                                 50)

        clicked_color = (155, 155, 155)
        color = (255, 255, 255)

        self._draw_button(surface, "Resume" if self.paused else "Pause",
                          pause_rect,
                          clicked_color if self.paused_clicked else color,
                          (0, 0, 0))

        # Draw a button to step through the simulation
        step_rect = pygame.Rect(x_offset,
                                y_offset + (button_row * row_height) + 110,
                                200,
                                50)
        self._draw_button(surface, "Step", step_rect,
                          clicked_color if self.step_clicked else color,
                          (0, 0, 0))

        return pause_rect, step_rect


    def render(self, grid_cells: np.ndarray, agent_pos: tuple[int, int],
               info: dict[str, any], reward: int, agent_storage_level: int, is_single_step: bool = False):
        """Render the environment.

        Args:
            grid_cells: The grid cells contained in the Grid class.
            agent_pos: List of current agent positions
            info: `info` dict held by the Environment class.
            is_single_step: Whether this render is caused by a click of the
                `step` button.
        """
        self.frame_count += 1
        self.frame_count %= 10

        curr_time = perf_counter()
        fps = 1 / (curr_time - self.last_render_time)
        self.last_render_time = curr_time

        self.last_10_fps[self.frame_count] = fps
        self.stats["fps"] = f"{sum(self.last_10_fps) / 10:.1f}"
        self.stats["total_targets_reached"] += int(info["target_reached"])

        if (not self.paused and not self.step) or is_single_step:
            self.stats["total_steps"] += 1

        failed_move = 1 - int(info["agent_moved"])
        self.stats["total_failed_move"] += failed_move

        self.stats["cumulative_reward"] += reward
        self.stats["agent_storage_level"] = agent_storage_level  # Update storage level

        # Create a surface to actually draw on
        background = pygame.Surface(self.window.get_size()).convert()
        background.fill((250, 250, 250))

        # Calculate grid offset
        grid_width = self.scalar * grid_cells.shape[0]
        grid_height = self.scalar * grid_cells.shape[1]
        x_offset = (self.grid_panel_size[0] / 2) - (grid_width / 2)
        y_offset = (self.grid_panel_size[1] / 2) - (grid_height / 2)

        # Draw the background for the info panel
        background.fill((238, 241, 240), self.info_panel_rect)
        # Draw each layer on the surface
        self._draw_grid(background, grid_cells, x_offset, y_offset)
        self._draw_agent(background, agent_pos, x_offset, y_offset)
        pause_rect, step_rect = self._draw_info(background)

        # Blit the surface onto the window
        update_rect = self.window.blit(background, background.get_rect())
        pygame.display.update(update_rect)

        # Parse events that happened since the last render step
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Detect click events
                if pause_rect.collidepoint(event.pos):
                    self.paused_clicked = True
                elif step_rect.collidepoint(event.pos):
                    self.step_clicked = True
            elif event.type == pygame.MOUSEBUTTONUP:
                # Only do the action on mouse button up, as is expected.
                if self.paused_clicked:
                    self.paused_clicked = False
                    self.paused = not self.paused
                elif self.step_clicked:
                    self.step_clicked = False
                    self.paused = True
                    self.step = True
        pygame.event.pump()

    @staticmethod
    def close():
        """Closes the UI."""
        pygame.quit()