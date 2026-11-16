"""Composite NEA rain radar overlays onto NEA's own base maps.

Pure Pillow, no Home Assistant imports. Everything here is blocking: call it from an
executor job. Ported from the ``singapore`` integration (kianhean/singapore-homeassistant).
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from PIL import Image, ImageDraw, ImageFont

NEA_MAP_URL = "https://www.nea.gov.sg/assets/images/map/{name}"


@dataclass(frozen=True)
class Basemap:
    """An NEA rain-areas base map and the part of the radar image it shows.

    ``crop`` is the (left, top, right, bottom) box, in pixels of data.gov.sg's
    480 x 480 radar PNG, that NEA's own overlay for this map shows; ``None`` means
    the whole image.
    """

    name: str
    crop: tuple[int, int, int, int] | None = None

    @property
    def url(self) -> str:
        return NEA_MAP_URL.format(name=self.name)


# Base maps used on https://www.nea.gov.sg/weather/rain-areas. The crop boxes were
# found by matching NEA's own overlays against data.gov.sg's image of the same
# capture: NEA's 70 km overlay (217 x 120) is exactly this crop, and its 240 km
# overlay is the full image. NEA has no map for the 480 km range.
BASEMAPS: Final[dict[str, Basemap]] = {
    "70km": Basemap("base-853.png", crop=(83, 193, 300, 313)),
    "240km": Basemap("240km-v2.jpg"),
}


# Animation timing: each frame, then a longer hold on the latest image before looping.
FRAME_MS: Final = 500
HOLD_MS: Final = 1500


def _draw_overlay(base: Image.Image, overlay: bytes, spec: Basemap) -> Image.Image:
    """Return a copy of ``base`` (RGBA) with the radar overlay drawn on it.

    NEA's rain-areas page draws both images at the same width, top-left aligned, so
    the overlay is scaled uniformly to the base map's width.
    """
    with Image.open(io.BytesIO(overlay)) as overlay_image:
        rain = overlay_image.convert("RGBA")
    if spec.crop is not None:
        rain = rain.crop(spec.crop)
    height = round(rain.height * base.width / rain.width)
    rain = rain.resize((base.width, height), Image.Resampling.BILINEAR)
    out = base.copy()
    out.alpha_composite(rain.crop((0, 0, base.width, min(height, base.height))))
    return out


def _open_basemap(basemap: bytes) -> Image.Image:
    with Image.open(io.BytesIO(basemap)) as base_image:
        return base_image.convert("RGBA")


def composite_radar(basemap: bytes, overlay: bytes, spec: Basemap) -> bytes:
    """Lay a radar overlay on its base map, the way NEA's rain-areas page does.

    Returns a JPEG.
    """
    image = _draw_overlay(_open_basemap(basemap), overlay, spec)
    out = io.BytesIO()
    image.convert("RGB").save(out, format="JPEG", quality=90)
    return out.getvalue()


def _label_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=size)
    except (TypeError, OSError):  # Pillow without FreeType: fixed bitmap font
        return ImageFont.load_default()


def _draw_label(image: Image.Image, text: str) -> None:
    """Draw ``text`` (the capture time) in a black box at the top left."""
    draw = ImageDraw.Draw(image)
    font = _label_font(max(14, image.width // 36))
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    margin, pad = 8, 6
    draw.rectangle(
        (
            margin,
            margin,
            margin + right - left + 2 * pad,
            margin + bottom - top + 2 * pad,
        ),
        fill=(0, 0, 0),
    )
    draw.text((margin + pad - left, margin + pad - top), text, font=font, fill="white")


def animate_radar(
    basemap: bytes, frames: Sequence[tuple[str, bytes]], spec: Basemap
) -> bytes:
    """Render ``(label, overlay PNG)`` frames, oldest first, as a looping GIF.

    Every frame is the overlay on the base map (as in ``composite_radar``) with its
    label, usually the capture time, at the top left. All frames share one palette,
    built from the whole loop, so the map doesn't flicker between frames.
    """
    if not frames:
        raise ValueError("No radar frames to animate")
    base = _open_basemap(basemap)
    images = []
    for label, overlay in frames:
        image = _draw_overlay(base, overlay, spec).convert("RGB")
        _draw_label(image, label)
        images.append(image)

    # Palette from a half-size stack of every frame: rain colours that appear in
    # only some frames still get palette entries.
    small = [image.reduce(2) for image in images]
    stack = Image.new("RGB", (small[0].width, small[0].height * len(small)))
    for index, image in enumerate(small):
        stack.paste(image, (0, index * image.height))
    palette = stack.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    quantized = [
        image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images
    ]

    out = io.BytesIO()
    quantized[0].save(
        out,
        format="GIF",
        save_all=True,
        append_images=quantized[1:],
        duration=[FRAME_MS] * (len(quantized) - 1) + [HOLD_MS],
        loop=0,
    )
    return out.getvalue()
