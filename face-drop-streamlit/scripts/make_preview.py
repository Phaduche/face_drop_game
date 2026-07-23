from __future__ import annotations

import base64
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "preview.png"
REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
sys.path.insert(0, str(ROOT))

from face_game import make_demo_portrait, prepare_face_game


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(BOLD if bold else REGULAR, size)


def decode_uri(uri: str) -> Image.Image:
    payload = uri.split(",", 1)[1]
    return Image.open(BytesIO(base64.b64decode(payload))).convert("RGBA")


def rounded_shadow(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    radius: int,
    offset: int,
    fill: str,
) -> None:
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow)
    shifted = (box[0] + offset, box[1] + offset, box[2] + offset, box[3] + offset)
    draw.rounded_rectangle(shifted, radius=radius, fill="#18171c")
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline="#18171c", width=2)
    canvas.alpha_composite(shadow)


def main() -> None:
    width, height = 1440, 1040
    canvas = Image.new("RGBA", (width, height), "#f7f5ef")

    glow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse((-190, -180, 510, 520), fill=(184, 167, 255, 78))
    gd.ellipse((1050, 610, 1580, 1140), fill=(255, 111, 174, 64))
    glow = glow.filter(ImageFilter.GaussianBlur(75))
    canvas.alpha_composite(glow)

    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((76, 62, 340, 108), radius=23, fill="#d9ff63", outline="#18171c", width=2)
    draw.text((96, 75), "OPENCV FACE PUZZLE", fill="#18171c", font=font(16, True))

    draw.text((68, 166), "PUT YOUR", fill="#18171c", font=font(84, True))
    draw.text((68, 252), "FACE", fill="#b8a7ff", font=font(124, True))
    draw.text((68, 372), "BACK.", fill="#18171c", font=font(104, True))
    draw.multiline_text(
        (76, 518),
        "Drag the scattered eyes, nose and mouth\n"
        "back to their original positions. Pieces\n"
        "snap into place when you get close.",
        fill=(24, 23, 28, 190),
        font=font(22),
        spacing=13,
    )

    chips = [("DRAG 4 PIECES", 76, 665, 226), ("TOUCH READY", 242, 665, 386), ("LIVE SCORE", 402, 665, 535)]
    for label, x1, y1, x2 in chips:
        draw.rounded_rectangle((x1, y1, x2, y1 + 40), radius=20, fill=(255, 255, 255, 180), outline=(24, 23, 28, 35), width=1)
        draw.text((x1 + 14, y1 + 11), label, fill="#18171c", font=font(13, True))

    draw.text((76, 766), "HOW TO PLAY", fill="#18171c", font=font(15, True))
    for idx, line in enumerate(("Grab a face piece", "Move it onto the face", "Beat your best time")):
        cy = 817 + idx * 55
        draw.ellipse((76, cy - 16, 108, cy + 16), fill="#ff6fae", outline="#18171c", width=1)
        draw.text((87, cy - 10), str(idx + 1), fill="#18171c", font=font(14, True))
        draw.text((124, cy - 12), line, fill="#18171c", font=font(19, True))

    card = (670, 46, 1378, 992)
    rounded_shadow(canvas, card, radius=32, offset=10, fill=(255, 255, 255, 230))
    draw = ImageDraw.Draw(canvas)
    draw.text((696, 72), "FACE DROP / 01", fill="#18171c", font=font(17, True))
    draw.ellipse((1254, 78, 1266, 90), fill="#ff6fae")
    draw.text((1276, 75), "READY", fill="#18171c", font=font(14, True))

    board_uri, pieces, _ = prepare_face_game(make_demo_portrait())
    board = decode_uri(board_uri).resize((642, 642), Image.Resampling.LANCZOS)

    mask = Image.new("L", board.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, 641, 641), radius=24, fill=255)
    board.putalpha(mask)
    canvas.alpha_composite(board, (703, 112))

    scale = 642 / 700
    for piece in pieces:
        image = decode_uri(piece["src"])
        pw = max(1, round(piece["w"] * scale))
        ph = max(1, round(piece["h"] * scale))
        image = image.resize((pw, ph), Image.Resampling.LANCZOS)
        angle = -piece["rotation"]
        image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
        px = round(703 + piece["startX"] * scale - (image.width - pw) / 2)
        py = round(112 + piece["startY"] * scale - (image.height - ph) / 2)
        shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
        shadow.putalpha(image.getchannel("A").filter(ImageFilter.GaussianBlur(8)))
        dark = Image.new("RGBA", image.size, (24, 23, 28, 95))
        dark.putalpha(shadow.getchannel("A").point(lambda value: value * 0.45))
        canvas.alpha_composite(dark, (px + 5, py + 9))
        canvas.alpha_composite(image, (px, py))

    y = 778
    buttons = [
        ("CHECK POSITION", (703, y, 987, y + 52), "#ff6fae"),
        ("HINT", (999, y, 1160, y + 52), "#ffffff"),
        ("RESET", (1172, y, 1345, y + 52), "#ffffff"),
    ]
    draw = ImageDraw.Draw(canvas)
    for label, box, color in buttons:
        draw.rounded_rectangle(box, radius=13, fill=color, outline="#18171c", width=2)
        text_box = draw.textbbox((0, 0), label, font=font(15, True))
        tw = text_box[2] - text_box[0]
        draw.text(((box[0] + box[2] - tw) / 2, box[1] + 16), label, fill="#18171c", font=font(15, True))

    draw.text((706, 857), "0 / 4 PLACED", fill="#18171c", font=font(14, True))
    draw.rounded_rectangle((1000, 862, 1194, 872), radius=5, fill="#ffffff", outline="#18171c", width=1)
    draw.text((1298, 857), "00:00", fill="#18171c", font=font(14, True))
    draw.text((702, 934), "Demo screen · Streamlit + OpenCV", fill=(24, 23, 28, 130), font=font(13))

    canvas.convert("RGB").save(OUT, quality=95)
    print(OUT)


if __name__ == "__main__":
    main()
