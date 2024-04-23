from functools import lru_cache
from typing import Any, Union

import matplotlib.font_manager
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

from morphocut.core import Node, Output, RawOrVariable, ReturnOutputs


@lru_cache(128)
def draw_scalebar(
    length_in_unit,
    px_per_unit=1,
    unit="px",
    mode="L",
    fg_color=0,
    bg_color=255,
    font=None,
    margin=10,
):
    length_px = round(length_in_unit * px_per_unit)

    h = 32
    w = length_px + 2 * margin
    img = PIL.Image.new(mode, (w, h), bg_color)

    d = PIL.ImageDraw.Draw(img)

    d.text((10, 5), f"{length_in_unit:.0f}{unit}", font=font, fill=fg_color)

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


@ReturnOutputs
@Output("image")
class DrawScalebar(Node):
    """Append a scalebar to an image.

    Args:
        font (str, PIL.ImageFont font, optional): The name of a font family or a PIL.ImageFont font.
            If not provided, a default font will be used.
    """

    def __init__(
        self,
        image: RawOrVariable[np.ndarray],
        length_in_unit,
        px_per_unit=1,
        unit="px",
        fg_color=0,
        bg_color=255,
        font: Union[None, str, Any] = "sans",
        margin=10,
    ):
        super().__init__()

        self.image = image

        if isinstance(font, str):
            font_fn = matplotlib.font_manager.FontManager().findfont(font)
            font = PIL.ImageFont.truetype(font_fn, 12)

        self.length_in_unit = length_in_unit
        self.px_per_unit = px_per_unit
        self.unit = unit
        self.fg_color = fg_color
        self.bg_color = bg_color
        self.font = font
        self.margin = margin

    def transform(
        self,
        image: np.ndarray,
        length_in_unit,
        px_per_unit,
        unit,
        fg_color,
        bg_color,
        font,
        margin,
    ):
        # TODO: Convert colors (see image.py)

        # Calculate an alpha-mask for the scalebar
        scalebar = draw_scalebar(
            length_in_unit=length_in_unit,
            px_per_unit=px_per_unit,
            unit=unit,
            mode="F",
            fg_color=1,
            bg_color=0,
            font=font,
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
