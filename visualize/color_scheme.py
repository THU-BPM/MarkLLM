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

# ===========================================
# color_scheme.py
# Description: Color scheme for visualization
# ===========================================

from matplotlib import pyplot as plt


class ColorScheme:
    """Color scheme for visualization."""

    def __init__(self, background_color='white', prefix_color='#D2B48C') -> None:
        """
            Initialize the color scheme.

            Parameters:
                background_color (str): The background color.
                prefix_color (str): The prefix color.
        """
        self.background_color = background_color
        self.prefix_color = prefix_color
    
    def set_background_color(self, color) -> None:
        self.background_color = color
    
    def set_prefix_color(self, color) -> None:
        self.prefix_color = color
    
    def get_legend_height(self, font_size):
        return font_size  


class ColorSchemeForDiscreteVisualization(ColorScheme):
    """Color scheme for discrete visualization (KGW Family)."""

    def __init__(self, background_color='white', prefix_color='#D2B48C', 
                 red_token_color='#EA9999', green_token_color='#B6D7A8') -> None:
        """
            Initialize the color scheme for discrete visualization.

            Parameters:
                background_color (str): The background color.
                prefix_color (str): The prefix color.
                red_token_color (str): The color for red tokens.
                green_token_color (str): The color for green tokens.
        """
        super().__init__(background_color, prefix_color)
        self.red_token_color = red_token_color
        self.green_token_color = green_token_color
    
    def set_red_token_color(self, color) -> None:
        self.red_token_color = color
    
    def set_green_token_color(self, color)-> None:
        self.green_token_color = color
    
    def get_legend_items(self):
        return [
            ("Green Token", self.green_token_color),
            ("Red Token", self.red_token_color),
            ("Ignored", self.prefix_color)
        ]


class ColorSchemeForContinuousVisualization(ColorScheme):
    """Color scheme for continuous visualization (Christ Family)."""

    def __init__(self, background_color='white', prefix_color='#D2B48C',
                 color_axis_name='viridis_r') -> None:
        """
            Initialize the color scheme for continuous visualization.

            Parameters:
                background_color (str): The background color.
                prefix_color (str): The prefix color.
                color_axis_name (str): The color axis name.
        """
        super().__init__(background_color, prefix_color)
        color_axis_name = color_axis_name
        self.color_axis = plt.get_cmap(color_axis_name)
    
    def set_color_axis(self, color_axis_name: str) -> None:
        self.color_axis = plt.get_cmap(color_axis_name)
    
    def get_color_from_axis(self, value: float) -> tuple:
        rgba_color = self.color_axis(value)
        rgba_color_int = tuple(int(255 * component) for component in rgba_color)
        return rgba_color_int[0], rgba_color_int[1], rgba_color_int[2]
    
    def get_legend_items(self):
        return [
            ("Prefix", self.prefix_color)
        ]