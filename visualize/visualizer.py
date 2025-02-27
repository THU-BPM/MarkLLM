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

# ====================================================
# visualizer.py
# Description: Implementation of watermark visualizers
# ====================================================

from typing import Union
from PIL import Image, ImageDraw
from .font_settings import FontSettings
from .legend_settings import LegendSettings, DiscreteLegendSettings, ContinuousLegendSettings
from .page_layout_settings import PageLayoutSettings
from .data_for_visualization import DataForVisualization
from .color_scheme import ColorScheme, ColorSchemeForDiscreteVisualization, ColorSchemeForContinuousVisualization


class BaseVisualizer:
    """Base class for visualizers."""

    def __init__(self, color_scheme: ColorScheme, font_settings: FontSettings, 
                 page_layout_settings: PageLayoutSettings, legend_settings: LegendSettings) -> None:
        """
            Initialize the visualizer.

            Parameters:
                color_scheme (ColorScheme): The color scheme.
                font_settings (FontSettings): The font settings.
                page_layout_settings (PageLayoutSettings): The page layout settings.
                legend_settings (LegendSettings): The legend settings.
        """
        self.color_scheme = color_scheme
        self.page_layout_settings = page_layout_settings
        self.font_settings = font_settings
        self.legend_settings = legend_settings

    
    def _should_visualize_weight(self, data: DataForVisualization, visualize_weight: bool) -> bool:
        """Check if weight visualization is needed."""
        return visualize_weight and data.weights is not None
    
    def _calculate_line_space(self, token: str, img_width: int, token_spacing: int, max_width: int) -> tuple:
        """Calculate new image width and check if it exceeds max width."""
        bbox = self.font_settings.font.getbbox(token)
        word_width = bbox[2] - bbox[0]
        
        new_img_width = img_width + word_width + token_spacing
        return new_img_width, new_img_width > max_width

    def _split_data_into_lines(self, data: DataForVisualization) -> tuple:
        """Splits tokens into lines based on the maximum width and calculates the total image height."""
        
        # Initialize variables
        lines, line, img_width = [], [], 0
        img_height = self.font_settings.font_size + self.page_layout_settings.line_spacing
        
        # Process each token and its highlight value
        for token, value in zip(data.decoded_tokens, data.highlight_values):
            img_width, should_break = self._calculate_line_space(token, img_width, 
                                                                self.page_layout_settings.token_spacing, 
                                                                self.page_layout_settings.max_width)
            
            # If the token exceeds the max width, start a new line
            if should_break:
                lines.append(line)
                line = []
                img_height += self.font_settings.font_size + self.page_layout_settings.line_spacing
                img_width, _ = self._calculate_line_space(token, 0, self.page_layout_settings.token_spacing, self.page_layout_settings.max_width)
            
            line.append((token, value))
        
        # Append the last line if it contains any tokens
        if line:
            lines.append(line)
        
        return lines, img_height
    
    def _get_weight_color(self, weight: float) -> str:
        """Get the color for a weight value where larger weights result in darker colors."""
        weight_color_scale = (255, 0)  
        shade = int(weight_color_scale[0] - (weight_color_scale[0] - weight_color_scale[1]) * weight)
        color = f"#{shade:02x}{shade:02x}{shade:02x}"
        return color

    def _highlight_single_token(self, draw: ImageDraw, token: str, value: Union[int, float, None], token_width: int, show_text: bool, x: int, y: int):
        """Highlight a token based on its value."""
        pass
    
    def _visualize_weight_for_a_token(self, draw: ImageDraw, weight: float, token_width: int, x: int, y: int):
        """Visualize the weight of a token as a colored bar."""

        # check if weight is within the range [0, 1], if not don't visualize
        if weight < 0 or weight > 1:
            return
        
        # Draw a colored bar to represent the weight
        weight_color = self._get_weight_color(weight)
        draw.rectangle([(x, y + self.font_settings.font_size + self.page_layout_settings.weight_bar_spacing), 
                        (x + token_width, y + self.font_settings.font_size + self.page_layout_settings.weight_bar_spacing + 
                         self.page_layout_settings.weight_bar_height)], fill=weight_color)
        
    def _display_legend(self, draw: ImageDraw, start_x: int, start_y: int, img_height: int):
        """Display the legend on the image."""
        pass

    def _display_weight_legend(self, draw, x, y):
        """Display the weight legend."""
        rect_width = self.legend_settings.rec_width
        text_offset = self.legend_settings.text_offset

        # Draw a weight bar
        weight_height = self.page_layout_settings.weight_bar_height
        draw.rectangle([x, y, x + rect_width, y + weight_height], fill=self._get_weight_color(0.8))
        draw.text((x + rect_width + text_offset, y + weight_height / 2 - self.legend_settings.legend_font_size / 2), 
                  "Weight", fill="black", font=self.legend_settings.legend_font) 

    def visualize(self, data: DataForVisualization, show_text=True, visualize_weight=True, display_legend=True):
        """Visualizes the data as an image with optional text and weight visualization."""
        
        # Determine if weight visualization is needed
        should_visualize_weight = self._should_visualize_weight(data, visualize_weight)

        # Calculate the height of each line
        line_height = self.font_settings.font_size + self.page_layout_settings.line_spacing

        # Split data into manageable lines
        lines, img_height = self._split_data_into_lines(data)

        # Compute the total dimensions of the image
        legend_width = self.legend_settings.legend_width if display_legend else 0
        total_width = (self.page_layout_settings.max_width + self.page_layout_settings.margin_l + 
                       self.page_layout_settings.margin_r + legend_width) 
        total_height = img_height + self.page_layout_settings.margin_t + self.page_layout_settings.margin_b

        # Create a new image with the specified background color
        img = Image.new('RGB', (total_width, total_height), color=self.color_scheme.background_color)

        # Prepare to draw on the image
        draw = ImageDraw.Draw(img)
        y = self.page_layout_settings.margin_t  # Initial y-coordinate for drawing

        # Draw each line of tokens
        for line in lines:
            x = self.page_layout_settings.margin_l  # Initial x-coordinate for each line
            for token, value in line:
                # Replace getsize with getbbox
                bbox = self.font_settings.font.getbbox(token)
                token_width = bbox[2] - bbox[0]

                # Highlight the token
                self._highlight_single_token(draw, token, value, token_width, show_text, x, y)

                # Optionally visualize the token's weight
                if should_visualize_weight:
                    weight = data.weights[data.decoded_tokens.index(token)]
                    self._visualize_weight_for_a_token(draw, weight, token_width, x, y)

                # Update x-coordinate for the next token
                x += token_width + self.page_layout_settings.token_spacing

            # Move to the next line
            y += line_height
        
        # Optionally display the legend
        if display_legend:
            x, y = self._display_legend(draw, self.page_layout_settings.max_width + self.page_layout_settings.margin_l + 
                                        self.page_layout_settings.margin_r, self.page_layout_settings.margin_t,
                                        img_height)
            if should_visualize_weight:
                self._display_weight_legend(draw, x, y)

        return img

class DiscreteVisualizer(BaseVisualizer):
    """Visualizer for discrete visualization (KGW Family)."""
    def __init__(self, 
                 color_scheme: ColorSchemeForDiscreteVisualization, 
                 font_settings: FontSettings, 
                 page_layout_settings: PageLayoutSettings,
                 legend_settings: DiscreteLegendSettings) -> None:
        """
            Initialize the discrete visualizer.

            Parameters:
                color_scheme (ColorSchemeForDiscreteVisualization): The color scheme.
                font_settings (FontSettings): The font settings.
                page_layout_settings (PageLayoutSettings): The page layout settings.
                legend_settings (DiscreteLegendSettings): The discrete legend settings.
        """
        super().__init__(color_scheme, font_settings, page_layout_settings, legend_settings)
    
    def _highlight_single_token(self, draw, token, value, token_width, show_text, x, y):
        """Highlight a token based on its value."""
        mapping = {
            0: self.color_scheme.red_token_color,
            1: self.color_scheme.green_token_color
        }

        if show_text:
            token_color = mapping.get(value, self.color_scheme.prefix_color)
            draw.text((x, y), token, fill=token_color, font=self.font_settings.font)
        else:
            token_color = mapping.get(value, self.color_scheme.prefix_color)
            draw.rectangle([(x, y), (x + token_width, y + self.font_settings.font_size)], fill=token_color)
    
    def _display_legend(self, draw, x, y, img_height=None):
        """Display the legend for discrete visualization."""
        items = self.color_scheme.get_legend_items()
        rect_width = self.legend_settings.rec_width
        rect_height = self.font_settings.font_size
        text_offset = self.legend_settings.text_offset

        # Draw each item in the legend
        y += self.legend_settings.top_spacing
        for label, color in items:
            draw.rectangle([x, y, x + rect_width, y + rect_height], fill=color)
            draw.text((x + rect_width + text_offset, y + rect_height / 2 - self.legend_settings.legend_font_size / 2), 
                      label, fill="black", font=self.legend_settings.legend_font)
            y += rect_height + text_offset

        return x, y


class ContinuousVisualizer(BaseVisualizer):
    """Visualizer for continuous visualization (Christ Family)."""
    def __init__(self, 
                 color_scheme: ColorSchemeForContinuousVisualization, 
                 font_settings: FontSettings, 
                 page_layout_settings: PageLayoutSettings,
                 legend_settings: ContinuousLegendSettings) -> None:
        """
            Initialize the continuous visualizer.

            Parameters:
                color_scheme (ColorSchemeForContinuousVisualization): The color scheme.
                font_settings (FontSettings): The font settings.
                page_layout_settings (PageLayoutSettings): The page layout settings.
                legend_settings (ContinuousLegendSettings): The continuous legend settings.
        """
        super().__init__(color_scheme, font_settings, page_layout_settings, legend_settings)
    
    def _highlight_single_token(self, draw, token, value, token_width, show_text, x, y):
        """Highlight a token based on its value."""
        if value is not None:
            color = self.color_scheme.get_color_from_axis(value)
        else:
            color = self.color_scheme.prefix_color

        if show_text:
            draw.text((x, y), token, fill=color, font=self.font_settings.font)
        else:
            draw.rectangle([(x, y), (x + token_width, y + self.font_settings.font_size)], fill=color)
    
    def _display_legend(self, draw, x, y, img_height):
        """Display the legend for continuous visualization."""
        axis_width = self.legend_settings.color_axis_width
        axis_offset = self.legend_settings.axis_offset
        num_ticks = self.legend_settings.axis_num_ticks

        # Draw the color axis with ticks
        for i in range(img_height):
            # Calculate color index from end to start
            color_index = 1 - i / img_height
            color = self.color_scheme.get_color_from_axis(color_index)
            draw.rectangle([x, y + i, x + axis_width, y + i + 1], fill=color)

        # Draw tick marks
        tick_spacing = img_height / (num_ticks - 1)
        for i in range(num_ticks):
            tick_y = y + i * tick_spacing
            draw.line([x + axis_width, tick_y, x + axis_width + 5, tick_y], fill="black", width=1)

            # Optionally, add labels for each tick
            label_value = 1 - i / (num_ticks - 1)  # Normalize the label
            label_text = f"{label_value:.2f}"  # Format the label text
            draw.text((x + axis_width + 10, tick_y - 7), label_text, fill="black", font=self.legend_settings.legend_font)

        # Update x-coordinate and y-coordinate for the next legend item
        x += axis_width + axis_offset
        y += self.legend_settings.top_spacing

        # Draw legend other than the color axis
        if not self.legend_settings.show_axis_only:
            items = self.color_scheme.get_legend_items()
            rect_width = self.legend_settings.rec_width
            rect_height = self.font_settings.font_size
            text_offset = self.legend_settings.text_offset
            for label, color in items:
                draw.rectangle([x, y, x + rect_width, y + rect_height], fill=color)
                draw.text((x + rect_width + text_offset, y + rect_height / 2 - self.legend_settings.legend_font_size / 2), 
                        label, fill="black", font=self.legend_settings.legend_font)
                y += rect_height + text_offset

        return x, y