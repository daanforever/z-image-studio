"""Placeholder image when the real pipeline is not loaded."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont


def wrap_text(text: str, font, max_width: int, draw: ImageDraw.ImageDraw) -> str:
    words = text.split()
    if not words:
        return text
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
            if len(lines) >= 10:
                current += "…"
                break
    lines.append(current)
    return "\n".join(lines[:11])


def demo_image(prompt: str, width: int, height: int, seed: int, reason: str) -> Image.Image:
    width = max(512, min(width, 1536))
    height = max(512, min(height, 1536))
    image = Image.new("RGB", (width, height), "#12100c")
    draw = ImageDraw.Draw(image)

    for y in range(height):
        t = y / max(height - 1, 1)
        r = int(18 + 28 * t)
        g = int(14 + 10 * t)
        b = int(8 + 4 * (1 - t))
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    accent = "#e8a54b"
    draw.rectangle([0, 0, 8, height], fill=accent)

    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 36)
        body_font = ImageFont.truetype("DejaVuSans.ttf", 22)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        title_font = body_font = small_font = ImageFont.load_default()

    margin = 48
    draw.text((margin, margin), "Z-Image-Turbo · demo", font=title_font, fill=accent)
    draw.text(
        (margin, margin + 56),
        reason or "Model not loaded — UI is running without weights.",
        font=small_font,
        fill="#c9bba8",
    )

    wrapped = wrap_text(prompt.strip() or "(empty prompt)", body_font, width - margin * 2, draw)
    draw.multiline_text((margin, margin + 110), wrapped, font=body_font, fill="#f4efe6", spacing=8)
    draw.text(
        (margin, height - 72),
        f"{width}×{height}   seed {seed}",
        font=small_font,
        fill="#8a7d6d",
    )
    return image
