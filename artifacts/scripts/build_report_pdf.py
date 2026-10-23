#!/usr/bin/env python3
"""Сборка PDF отчёта: reports/report.md → report.html → report.pdf.

Пайплайн: python-markdown (extensions: tables, fenced_code, sane_lists) →
HTML с печатным CSS (A4, поля, разрывы страниц) → headless Chrome
--print-to-pdf. Картинки (скриншот Kaggle) подхватываются относительно
reports/, потому что HTML пишется рядом с report.md.

Запуск:
  uv run --with markdown python artifacts/scripts/build_report_pdf.py
Выход: reports/report.html, reports/report.pdf.
"""
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
MD = REPORTS / "report.md"
HTML = REPORTS / "report.html"
PDF = REPORTS / "report.pdf"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

CSS = """
@page { size: A4; margin: 17mm 15mm 16mm 15mm; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
  font-size: 10.3pt; line-height: 1.45; color: #1a1a1a;
  max-width: 100%;
}
h1 { font-size: 19pt; line-height: 1.25; margin: 0 0 10pt 0;
     border-bottom: 2px solid #333; padding-bottom: 6pt; }
h2 { font-size: 14pt; margin: 18pt 0 8pt 0; padding-top: 4pt;
     border-top: 1px solid #ccc; }
h2:first-of-type { border-top: none; }
h3 { font-size: 11.5pt; margin: 12pt 0 5pt 0; }
h4 { font-size: 10.5pt; margin: 10pt 0 4pt 0; }
p, ul, ol { margin: 4pt 0 7pt 0; }
ul, ol { padding-left: 18pt; }
li { margin: 2pt 0; }
code { font-family: Menlo, "SF Mono", monospace; font-size: 8.6pt;
       background: #f2f2f2; padding: 0.5pt 2.5pt; border-radius: 2pt; }
pre { background: #f6f6f6; border: 1px solid #ddd; border-radius: 3pt;
      padding: 7pt 9pt; overflow-wrap: break-word; white-space: pre-wrap;
      page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8.4pt; }
table { border-collapse: collapse; width: 100%; margin: 7pt 0 10pt 0;
        font-size: 8.7pt; page-break-inside: auto; }
tr { page-break-inside: avoid; }
th, td { border: 0.6pt solid #bbb; padding: 2.8pt 5pt; text-align: left;
         vertical-align: top; }
th { background: #ececec; font-weight: 600; }
td[style*="right"], th[style*="right"] { text-align: right; }
img { max-width: 100%; border: 1px solid #ccc; border-radius: 3pt;
      page-break-inside: avoid; }
blockquote { margin: 7pt 0; padding: 5pt 10pt; border-left: 3pt solid #999;
             background: #f7f7f7; color: #333; }
a { color: #144d8a; text-decoration: none; }
hr { border: none; border-top: 1px solid #ccc; margin: 14pt 0; }
strong { font-weight: 650; }
"""

HTML_TMPL = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<title>Отчёт: sweettv-tv-program-recommender</title>
<style>{css}</style></head>
<body>{body}</body></html>
"""


def main() -> None:
    if not MD.exists():
        sys.exit(f"[error] нет {MD}")
    if not CHROME.exists():
        sys.exit(f"[error] не найден Chrome: {CHROME}")

    md_text = MD.read_text(encoding="utf-8")
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )
    HTML.write_text(HTML_TMPL.format(css=CSS, body=body), encoding="utf-8")
    print(f"[html] {HTML} ({len(body)} символов тела)")

    cmd = [
        str(CHROME),
        "--headless", "--disable-gpu", "--hide-scrollbars",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF}",
        f"file://{HTML}",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if res.returncode != 0:
        sys.exit(f"[error] chrome rc={res.returncode}\n{res.stderr[-1500:]}")
    if not PDF.exists():
        sys.exit("[error] PDF не создан")
    print(f"[pdf] {PDF} ({PDF.stat().st_size} байт)")


if __name__ == "__main__":
    main()
