from io import BytesIO

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from pathlib import Path


def get_path_to_figures():
    """
    Returns the path where the processed data is stored

    Parameters:
        ----------
        None

    Returns:
        -------
        path_to_processed_data: str (Path)
            Path format string

    """
    path_to_figure = Path.cwd() / 'indentation\\model\\comsol\\sensitivity_analysis\\figures'
    return path_to_figure

class Fonts:
    def serif(self):
        font = font_manager.FontProperties(family="serif", weight="normal", style="normal", size=18)
        return font
    
    def serif_1(self):
        font = font_manager.FontProperties(family="serif", weight="normal", style="normal", size=24)
        return font
    
    def serif_rz_legend(self):
        font = font_manager.FontProperties(family="serif", weight="normal", style="normal", size=9)
        return font
    
    def serif_3horizontal(self):
        font = font_manager.FontProperties(family="serif", weight="normal", style="normal", size=40)
        return font

    def serif_horizontalfigure(self):
        font = font_manager.FontProperties(family="serif", weight="normal", style="normal", size=24)
        return font

    def axis_legend_size(self):
        return 24

    def axis_label_size(self):
        return 25

    def fig_title_size(self):
        return 18
    
    
class SaveFigure:
    def save_as_png(self, fig, filename):
        filename_png = filename + '.png'
        fig.savefig(get_path_to_figures() / filename_png, format="png")

    def save_as_svg(self, fig, filename):
        filename_svg = filename + '.svg'
        fig.savefig(get_path_to_figures() / filename_svg, format="svg")
        
        
        
class CreateFigure:
    def rectangle_figure(self, pixels):
        fig = plt.figure(figsize=(9, 6), dpi=pixels, constrained_layout=True)
        return fig

    def rectangle_rz_figure(self, pixels):
        fig = plt.figure(figsize=(9, 4), dpi=pixels, constrained_layout=True)
        return fig

    def square_figure(self, pixels):
        fig = plt.figure(figsize=(6, 6), dpi=pixels, constrained_layout=True)
        return fig

    def square_figure_7(self, pixels):
        fig = plt.figure(figsize=(7, 7), dpi=pixels, constrained_layout=True)
        return fig
    
    def square_figure_10(self, pixels):
        fig = plt.figure(figsize=(10, 10), dpi=pixels, constrained_layout=True)
        return fig
    
    def rectangle_vertical_rz_figure(self, pixels):
        fig = plt.figure(figsize=(6, 9), dpi=pixels, constrained_layout=True)
        return fig


