# ============================================
# font_settings.py
# Description: Font settings for visualization
# ============================================

from PIL import ImageFont


class FontSettings:
    """Font settings for visualization."""

    def __init__(self, font_path: str = "font/Courier_New_Bold.ttf", font_size: int = 20) -> None:
        """
            Initialize the font settings.

            Parameters:
                font_path (str): The path to the font file.
                font_size (int): The font size.
        """
        self.font_path = font_path
        self.font_size = font_size
        self.font = ImageFont.truetype(self.font_path, self.font_size)
        