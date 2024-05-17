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