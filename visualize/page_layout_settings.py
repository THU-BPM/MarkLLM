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

# ===================================================
# page_layout_settings.py
# Description: Page layout settings for visualization
# ===================================================


class PageLayoutSettings:
    """Page layout settings for visualization."""

    def __init__(self, token_spacing: int = 8, max_width: int = 400, line_spacing: int = 18, 
                  weight_bar_height: int = 5, weight_bar_spacing: int = 3,
                  margin_t: int = 10, margin_b: int = 10, margin_l: int = 10, margin_r: int = 10) -> None:
        """
            Initialize the page layout settings.

            Parameters:
                token_spacing (int): The token spacing.
                max_width (int): The maximum width.
                line_spacing (int): The line spacing.
                margin_t (int): The top margin.
                margin_b (int): The bottom margin.
                margin_l (int): The left margin.
                margin_r (int): The right margin.
                weight_bar_height (int): The weight bar height.
                weight_bar_spacing (int): The weight bar spacing.
        """
        self.token_spacing = token_spacing
        self.max_width = max_width
        self.line_spacing = line_spacing
        self.margin_t = margin_t
        self.margin_b = margin_b
        self.margin_l = margin_l
        self.margin_r = margin_r
        self.weight_bar_height = weight_bar_height
        self.weight_bar_spacing = weight_bar_spacing