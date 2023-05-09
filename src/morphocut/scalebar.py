from functools import lru_cache

import matplotlib.font_manager
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

from morphocut.core import Node, Output, RawOrVariable, ReturnOutputs


@lru_cache(128)
def draw_scalebar(
    length_unit,
    px_per_unit=1,
    unit="px",
    mode="L",
    fg_color=0,
    bg_color=255,
    font_family="sans",
    margin=10,
):
    """
       Draw a scalebar image of specified length and units.

       Args:
           length_unit: The length of the scalebar in the specified unit.
           px_per_unit: The ratio of pixels to units for the scalebar.
           unit: The unit of length for the scalebar.
           mode: The color mode to use for the PIL image.
           fg_color: The color to use for the scalebar and text.
           bg_color: The color to use for the background.
           font_family: The font family to use for the scalebar text.
           margin: The margin to use around the scalebar.

       Returns:
           A numpy array representing the image of the scalebar.
       """

    length_px = round(length_unit * px_per_unit)

    h = 32
    w = length_px + 2 * margin
    img = PIL.Image.new(mode, (w, h), bg_color)

    font_fn = matplotlib.font_manager.FontManager().findfont(font_family)
    fnt = PIL.ImageFont.truetype(font_fn, 12)
    d = PIL.ImageDraw.Draw(img)

    d.text((10, 5), f"{length_unit:.0f}{unit}", font=fnt, fill=fg_color)

    d.line(
        [
            (margin, 28),
            (margin, 25),
            (margin + length_px, 25),
            (margin + length_px, 28),
        ],
        fill=fg_color,
    )

    return np.asarray(img)


# Commit message: Fixes #103
@ReturnOutputs
@Output("image")
class DrawScalebar(Node):
    """
    Append a scalebar to an image.

       Args:
           image: The image to append the scalebar to.
           length_unit: The length of the scalebar in the specified unit.
           px_per_unit: The ratio of pixels to units for the scalebar.
           unit: The unit of length for the scalebar.
           fg_color: The color to use for the scalebar and text.
           bg_color: The color to use for the background.
           font_family: The font family to use for the scalebar text.
           margin: The margin to use around the scalebar.
       """

    def __init__(
        self,
        image: RawOrVariable[np.ndarray],
        length_unit,
        px_per_unit=1,
        unit="px",
        fg_color=0,
        bg_color=255,
        font_family="sans",
        margin=10,
    ):
        super().__init__()

        self.image = image

        self.length_unit = length_unit
        self.px_per_unit = px_per_unit
        self.unit = unit
        self.fg_color = fg_color
        self.bg_color = bg_color
        self.font_family = font_family
        self.margin = margin

    def transform(
        self,
        image: np.ndarray,
        length_unit,
        px_per_unit,
        unit,
        fg_color,
        bg_color,
        font_family,
        margin,
    ):
        # TODO: Convert colors (see image.py)

        # Calculate an alpha-mask for the scalebar
        scalebar = draw_scalebar(
            length_unit=length_unit,
            px_per_unit=px_per_unit,
            unit=unit,
            mode="F",
            fg_color=1,
            bg_color=0,
            font_family=font_family,
            margin=margin,
        )

        # Construct canvas that can contain the image and the scalebar
        cheight = image.shape[0] + scalebar.shape[0]
        cwidth = max(image.shape[1], scalebar.shape[1])
        canvas = np.full(
            (cheight, cwidth) + image.shape[2:], bg_color, dtype=image.dtype
        )

        # Paste image (centered)
        offs = (cwidth - image.shape[1]) // 2
        canvas[: image.shape[0], offs : offs + image.shape[1]] = image

        # Paste scalebar (aligned left)
        canvas[
            image.shape[0] : image.shape[0] + scalebar.shape[0], : scalebar.shape[1]
        ] = (scalebar * fg_color) + (1 - scalebar) * bg_color

        return canvas
