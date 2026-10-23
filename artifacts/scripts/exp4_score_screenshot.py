#!/usr/bin/env python3
r"""exp4: PNG-скриншот возвращённого Kaggle-скора из вывода CLI.

Вход: stdout команды `kaggle competitions submissions -c sweettv-...`
(ссылка на pipe: stdin). Выход: reports/kaggle_public_score_final.png —
изображение таблицы сабмитов, где видна строка назначенного финала
(exp4_k10, run-id 57554720, publicScore 0.34586).

Использование:
  eval "$(grep -E '^\s*(export\s+)?KAGGLE_API_TOKEN=' ~/.zshrc | tail -1)"
  kaggle competitions submissions -c sweettv-tv-program-recommender
    | uv run --with pillow python artifacts/scripts/exp4_score_screenshot.py
      reports/kaggle_public_score_final.png
"""
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# терминальная тема: тёмный фон, светлый текст; финальная строка — жёлтая метка ★
BG = (30, 30, 30)
FG = (220, 220, 220)
DIM = (150, 150, 150)
HL = (255, 200, 60)
CMD = (120, 200, 120)

PAD = 24
LINE_H = 20
FONT_SIZE = 14


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: exp4_score_screenshot.py <out.png>")
    out_path = Path(sys.argv[1])
    if out_path.exists():
        sys.exit(f"[refuse] {out_path} уже существует — не перезаписываю")
    raw = sys.stdin.read()
    rows = [ln.rstrip() for ln in raw.splitlines()
            if ln.strip() and not ln.startswith("Warning:")]
    if not rows:
        sys.exit("[error] пустой вход — нет таблицы сабмитов")
    if not any("57554720" in ln for ln in rows):
        sys.exit("[error] в выводе нет строки финала 57554720")

    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Menlo.ttc", FONT_SIZE)
        font_b = ImageFont.truetype(
            "/System/Library/Fonts/Menlo.ttc", FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()
        font_b = font

    header = [
        "$ kaggle competitions submissions -c sweettv-tv-program-recommender",
        f"(снято {datetime.now():%Y-%m-%d %H:%M} локально; команда возвращает "
        "public/private сразу после COMPLETE)",
        "",
    ]
    lines = header + rows + [""]

    # ширина по самой длинной строке (моноширинный шрифт)
    sample_w = max(font.getlength(ln) for ln in lines)
    width = int(sample_w) + 2 * PAD
    height = LINE_H * len(lines) + 2 * PAD + 30

    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    y = PAD
    d.text((PAD, y), lines[0], font=font_b, fill=CMD)
    y += LINE_H
    d.text((PAD, y), lines[1], font=font, fill=DIM)
    y += LINE_H
    y += LINE_H  # пустая строка header
    for ln in lines[3:]:
        fill = HL if "57554720" in ln else (
            FG if not ln.startswith("---") and not ln.startswith("ref") else DIM)
        d.text((PAD, y), ln, font=font, fill=fill)
        y += LINE_H
    d.text((PAD, height - PAD - 6),
           "sweet.tv — TV program recommender | kaggle competitions submissions",
           font=font, fill=DIM)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"[saved] {out_path} ({width}x{height})")


if __name__ == "__main__":
    main()
