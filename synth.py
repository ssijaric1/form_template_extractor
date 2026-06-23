"""
synth.py — generate synthetic filled attendance registers + simulate phone scans.

Why this exists
---------------
The real dataset (~200 phone scans) may not be available while developing, and
a ground-truth blank template is needed to sanity-check the extractor anyway.
This module renders forms that mimic the reference register
(crest, title, handwritten date, 34-row table with NUMBER / TIME / NAME /
DEPARTMENT / SIGNATURE columns), fills a random number of rows with fake
"handwriting", and then degrades each page the way a phone camera would:
perspective tilt on a desk, rotation, uneven lighting, blur, sensor noise,
JPEG compression.

Names, departments and signatures are randomly generated — no real people.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# setting page geometry (A4 at ~150 dpi):
PAGE_W, PAGE_H = 1240, 1754
TABLE_X0, TABLE_X1 = 100, 1150
TABLE_Y0 = 300
HEADER_H, ROW_H, N_ROWS = 44, 38, 34
COLS = [("NUMBER", 90), ("TIME", 90), ("NAME", 350),
        ("DEPARTMENT", 230), ("SIGNATURE", 290)]

TITLE = "OFFICIALS ORDINARY COUNCIL REGISTER"

_SURNAMES = ["Marenga", "Chideme", "Kotani", "Mufaro", "Sibanda", "Dube",
             "Moyo", "Ncube", "Phiri", "Zulu", "Tembo", "Chirwa", "Mwale",
             "Banda", "Masuku", "Nyathi", "Gumbo", "Shumba", "Mlambo",
             "Chikore", "Mutsvairo", "Hove", "Makoni", "Zinyama"]
_INITIALS = list("ABCDEFGHJKLMNPRSTVW")
_DEPTS = ["C/S", "TC", "HCD", "A/DLW", "MLG PW", "ATC",
          "ADHCS", "TC (Audit)", "APO", "MLG&PW"]
_INK_COLORS = [(15, 15, 15), (25, 25, 95), (20, 30, 120), (40, 40, 40)]

_FONT_CANDIDATES = {
    False: ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf"],
    True: ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
           "C:/Windows/Fonts/arialbd.ttf",
           "/System/Library/Fonts/Supplemental/Arial Bold.ttf"],
}


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES[bold]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:                                   # using Pillow >= 10's scalable default:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()



def hand_text(img: Image.Image, xy: tuple[float, float], text: str,
              size: int, rng: random.Random, color: tuple[int, int, int]):
    """Draw text one jittered, slightly rotated character at a time."""
    draw = ImageDraw.Draw(img)
    x, y = xy
    for ch in text:
        s = max(8, int(size * rng.uniform(0.9, 1.12)))
        font = get_font(s)
        if ch == " ":
            x += s * 0.45
            continue
        w = draw.textlength(ch, font=font)
        tile = Image.new("RGBA", (int(w) + 10, int(s * 1.6) + 10), (0, 0, 0, 0))
        ImageDraw.Draw(tile).text((5, 3), ch, font=font, fill=color + (255,))
        tile = tile.rotate(rng.uniform(-9, 9), expand=True,
                           resample=Image.BICUBIC)
        img.paste(tile, (int(x), int(y + rng.uniform(-2.5, 2.5))), tile)
        x += w * rng.uniform(0.85, 1.05)


def signature_squiggle(draw: ImageDraw.ImageDraw, box, rng: random.Random,
                       color: tuple[int, int, int]):
    """A random cursive-looking flourish inside `box` = (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    x = x0 + rng.uniform(0.02, 0.15) * w
    y = y0 + h * rng.uniform(0.35, 0.65)
    vy = 0.0
    pts = [(x, y)]
    for _ in range(rng.randint(10, 16)):
        vy = vy * 0.45 + rng.uniform(-0.38, 0.38) * h
        x += rng.uniform(0.04, 0.11) * w
        y = min(max(y + vy, y0 + 2), y1 - 2)
        pts.append((x, y))
        if x > x1 - 6:
            break
    draw.line(pts, fill=color, width=rng.choice([2, 2, 3]), joint="curve")
    if rng.random() < 0.5:
        cx, cy = pts[rng.randrange(len(pts))]
        r = rng.uniform(4, 9)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     outline=color, width=2)



def _draw_crest(draw: ImageDraw.ImageDraw):
    """Simple stand-in for the coat of arms in the corner."""
    x, y = 95, 35
    draw.ellipse([x, y, x + 95, y + 75], outline=(0, 0, 0), width=3)
    draw.polygon([(x + 18, y + 70), (x + 77, y + 70), (x + 47, y + 125)],
                 outline=(0, 0, 0), width=3)
    for dx in (28, 47, 66):
        draw.line([(x + dx, y + 15), (x + dx, y + 60)], fill=(0, 0, 0), width=3)
    draw.line([(x + 8, y + 130), (x + 87, y + 130)], fill=(0, 0, 0), width=2)


def make_filled_form(rng: random.Random) -> Image.Image:
    img = Image.new("RGB", (PAGE_W, PAGE_H), (252, 252, 250))
    draw = ImageDraw.Draw(img)

    _draw_crest(draw)
    title_font = get_font(30, bold=True)
    tw = draw.textlength(TITLE, font=title_font)
    tx = (PAGE_W - tw) / 2 + 40
    draw.text((tx, 80), TITLE, font=title_font, fill=(0, 0, 0))
    draw.line([(tx, 118), (tx + tw, 118)], fill=(0, 0, 0), width=2)

    xs = [TABLE_X0]
    for _, w in COLS:
        xs.append(xs[-1] + w)
    y_top, y_head = TABLE_Y0, TABLE_Y0 + HEADER_H
    y_bot = y_head + N_ROWS * ROW_H

    draw.rectangle([TABLE_X0, y_top, xs[-1], y_bot], outline=(0, 0, 0), width=2)
    head_font, num_font = get_font(20, bold=True), get_font(16)
    for (label, _), xa, xb in zip(COLS, xs[:-1], xs[1:]):
        lw = draw.textlength(label, font=head_font)
        draw.text((xa + (xb - xa - lw) / 2, y_top + 11), label,
                  font=head_font, fill=(0, 0, 0))
    for x in xs[1:-1]:
        draw.line([(x, y_top), (x, y_bot)], fill=(0, 0, 0), width=1)
    draw.line([(TABLE_X0, y_head), (xs[-1], y_head)], fill=(0, 0, 0), width=2)
    for r in range(1, N_ROWS):
        y = y_head + r * ROW_H
        draw.line([(TABLE_X0, y), (xs[-1], y)], fill=(0, 0, 0), width=1)
    for r in range(N_ROWS):
        y = y_head + r * ROW_H
        draw.text((TABLE_X0 + 12, y + 9), str(r + 1),
                  font=num_font, fill=(0, 0, 0))

    ink = rng.choice(_INK_COLORS)
    date = f"{rng.randint(1, 28)}/{rng.randint(1, 12)}/2{rng.randint(2, 5)}"
    hand_text(img, (PAGE_W - 290, 50), date, 42, rng, ink)
    draw.line([(PAGE_W - 305, 105), (PAGE_W - 95, 100)], fill=ink, width=2)

    n_filled = rng.randint(10, min(24, N_ROWS))
    for r in range(N_ROWS):
        if r < n_filled:
            if rng.random() < 0.10:
                continue
        elif rng.random() > 0.06:
            continue
        y = y_head + r * ROW_H
        row_ink = rng.choice(_INK_COLORS) if rng.random() < 0.3 else ink
        sep = rng.choice([". ", " ", ".", ".  "])
        name = (f"{rng.choice(_INITIALS)}{sep}"
                f"{rng.choice(_SURNAMES)}")
        hand_text(img, (xs[2] + rng.uniform(6, 120), y + rng.uniform(2, 7)),
                  name, rng.randint(19, 24), rng, row_ink)
        hand_text(img, (xs[3] + rng.uniform(8, 90), y + rng.uniform(3, 8)),
                  rng.choice(_DEPTS), rng.randint(17, 21), rng, row_ink)
        signature_squiggle(draw, (xs[4] + 8, y + 4, xs[5] - 8, y + ROW_H - 4),
                           rng, row_ink)
        if rng.random() < 0.15:
            hand_text(img, (xs[1] + 8, y + 6), f"8:{rng.randint(0, 5)}0",
                      18, rng, row_ink)
    return img



def scan_simulate(pil_img: Image.Image, rng: random.Random,
                  nprng: np.random.Generator,
                  pristine: bool = False) -> bytes:
    """Render a synthetic phone scan.

    When ``pristine=True``, the page is laid down upright with no rotation,
    no perspective jitter, no lighting gradient, and no blur — only mild
    sensor noise and JPEG. These scans win the sharpness ranking and get
    picked as the alignment reference, so the extracted template comes out
    upright.
    """
    page = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    ph, pw = page.shape[:2]
    pad = int(0.05 * pw)
    cw, ch = pw + 2 * pad, ph + 2 * pad

    desk = 245 if pristine else rng.randint(60, 115)
    gy, gx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    if pristine:
        canvas = np.full((ch, cw, 3), desk, np.uint8)
    else:
        grad = (desk + 22 * (gx / cw - 0.5) * rng.uniform(-1, 1)
                     + 22 * (gy / ch - 0.5) * rng.uniform(-1, 1))
        canvas = np.clip(grad, 0, 255).astype(np.uint8)[..., None].repeat(3, 2)

    j = 0.0 if pristine else 0.030 * pw
    src = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
    dst = np.float32([[pad + rng.uniform(-j, j), pad + rng.uniform(-j, j)]
                      for _ in range(4)])
    dst += src
    dst[:, 0] = np.clip(dst[:, 0], 2, cw - 3)
    dst[:, 1] = np.clip(dst[:, 1], 2, ch - 3)
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(page, H, (cw, ch), borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), H, (cw, ch),
                               borderValue=0)
    canvas[mask > 0] = warped[mask > 0]

    ang = 0.0 if pristine else rng.uniform(-2.2, 2.2)
    if ang != 0.0:
        M = cv2.getRotationMatrix2D((cw / 2, ch / 2), ang, 1.0)
        canvas = cv2.warpAffine(canvas, M, (cw, ch),
                                borderValue=(desk, desk, desk))

    if not pristine:
        ax, ay = rng.uniform(-0.18, 0.18), rng.uniform(-0.18, 0.18)
        plane = rng.uniform(0.85, 1.04) * (1 + ax * (gx / cw - 0.5)
                                             + ay * (gy / ch - 0.5))
        canvas = np.clip(canvas.astype(np.float32) * plane[..., None],
                         0, 255).astype(np.uint8)

    if not pristine and rng.random() < 0.5:
        canvas = cv2.GaussianBlur(canvas, (3, 3), rng.uniform(0.3, 0.9))
    noise_sigma = 1.0 if pristine else rng.uniform(2.5, 6.5)
    noise = nprng.normal(0, noise_sigma, canvas.shape).astype(np.float32)
    canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    q = 95 if pristine else rng.randint(58, 88)
    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()



def make_dataset_bytes(n: int = 12, seed: int = 42,
                       n_upright: int = 2) -> list[tuple[str, bytes]]:
    """Return n synthetic scans as (filename, jpeg_bytes) pairs.

    The first ``n_upright`` scans are pristine (no rotation/perspective/blur)
    so the pipeline picks one of them as the alignment reference and the
    extracted template comes out upright.
    """
    out = []
    n_upright = max(1, min(int(n_upright), n))
    for i in range(n):
        rng = random.Random(seed * 9973 + i)
        nprng = np.random.default_rng(seed * 1000 + i)
        img = make_filled_form(rng)
        out.append((f"form_{i + 1:03d}.jpg",
                    scan_simulate(img, rng, nprng, pristine=i < n_upright)))
    return out


def make_ground_truth_blank(seed: int = 42) -> Image.Image:
    """The clean printed template (no handwriting) for visual comparison."""
    rng = random.Random(seed)

    class _NoFill(random.Random):
        pass

    img = Image.new("RGB", (PAGE_W, PAGE_H), (252, 252, 250))
    draw = ImageDraw.Draw(img)
    _draw_crest(draw)
    title_font = get_font(30, bold=True)
    tw = draw.textlength(TITLE, font=title_font)
    tx = (PAGE_W - tw) / 2 + 40
    draw.text((tx, 80), TITLE, font=title_font, fill=(0, 0, 0))
    draw.line([(tx, 118), (tx + tw, 118)], fill=(0, 0, 0), width=2)
    xs = [TABLE_X0]
    for _, w in COLS:
        xs.append(xs[-1] + w)
    y_top, y_head = TABLE_Y0, TABLE_Y0 + HEADER_H
    y_bot = y_head + N_ROWS * ROW_H
    draw.rectangle([TABLE_X0, y_top, xs[-1], y_bot], outline=(0, 0, 0), width=2)
    head_font, num_font = get_font(20, bold=True), get_font(16)
    for (label, _), xa, xb in zip(COLS, xs[:-1], xs[1:]):
        lw = draw.textlength(label, font=head_font)
        draw.text((xa + (xb - xa - lw) / 2, y_top + 11), label,
                  font=head_font, fill=(0, 0, 0))
    for x in xs[1:-1]:
        draw.line([(x, y_top), (x, y_bot)], fill=(0, 0, 0), width=1)
    draw.line([(TABLE_X0, y_head), (xs[-1], y_head)], fill=(0, 0, 0), width=2)
    for r in range(1, N_ROWS):
        y = y_head + r * ROW_H
        draw.line([(TABLE_X0, y), (xs[-1], y)], fill=(0, 0, 0), width=1)
    for r in range(N_ROWS):
        y = y_head + r * ROW_H
        draw.text((TABLE_X0 + 12, y + 9), str(r + 1),
                  font=num_font, fill=(0, 0, 0))
    return img


def save_dataset(out_dir: str | Path, n: int = 12, seed: int = 42) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, data in make_dataset_bytes(n, seed):
        p = out_dir / name
        p.write_bytes(data)
        paths.append(p)
    return paths


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    dest = sys.argv[2] if len(sys.argv) > 2 else "synthetic_forms"
    files = save_dataset(dest, n=n)
    print(f"Wrote {len(files)} synthetic scans to {dest}/")
