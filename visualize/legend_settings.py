# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ===============================================
# legend_settings.py
# Description: Legend settings for visualization
# ===============================================

from PIL import ImageFont


class LegendSettings:
    """Legend settings for visualization."""

    def __init__(self, legend_font_path: str = "font/arial.ttf", legend_font_size: int = 12, legend_width: int = 160) -> None:
        """
            Initialize the legend settings.

            Parameters:
                legend_font_path (str): The path to the legend font file.
                legend_font_size (int): The legend font size.
                legend_width (int): The legend width.
        """
        self.legend_font_path = legend_font_path
        self.legend_font_size = legend_font_size
        self.legend_font = ImageFont.truetype(self.legend_font_path, self.legend_font_size)
        self.legend_width = legend_width


class DiscreteLegendSettings(LegendSettings):
    """Legend settings for discrete visualization."""

    def __init__(self, legend_font_path: str = "font/arial.ttf", legend_font_size: int = 12, legend_width: int = 160,
                 rec_width: int = 50, text_offset: int = 20, top_spacing: int = 20) -> None:
        """
            Initialize the discrete legend settings.

            Parameters:
                legend_font_path (str): The path to the legend font file.
                legend_font_size (int): The legend font size.
                legend_width (int): The legend width.
                rec_width (int): The rectangle width.
                text_offset (int): The text offset.
                top_spacing (int): The top spacing.
        """
        super().__init__(legend_font_path, legend_font_size, legend_width)
        self.rec_width = rec_width
        self.text_offset = text_offset
        self.top_spacing = top_spacing


class ContinuousLegendSettings(LegendSettings):
    """Legend settings for continuous visualization."""

    def __init__(self, legend_font_path: str = "font/arial.ttf", legend_font_size: int = 12, legend_width: int = 160,
                 rec_width: int = 50, text_offset: int = 20, top_spacing: int = 20, axis_offset: int = 20, color_axis_width: int = 20, 
                 axis_num_ticks: int = 5, show_axis_only: bool = True) -> None:
        """
            Initialize the continuous legend settings.

            Parameters:
                legend_font_path (str): The path to the legend font file.
                legend_font_size (int): The legend font size.
                legend_width (int): The legend width.
                rec_width (int): The rectangle width.
                text_offset (int): The text offset.
                top_spacing (int): The top spacing.
                axis_offset (int): The axis offset.
                color_axis_width (int): The color axis width.
                axis_num_ticks (int): The number of ticks on the axis.
                show_axis_only (bool): Whether to show the axis only.
        """
        super().__init__(legend_font_path, legend_font_size, legend_width)
        self.rec_width = rec_width
        self.text_offset = text_offset
        self.top_spacing = top_spacing
        self.axis_offset = axis_offset
        self.color_axis_width = color_axis_width
        self.axis_num_ticks = axis_num_ticks
        self.show_axis_only = show_axis_only