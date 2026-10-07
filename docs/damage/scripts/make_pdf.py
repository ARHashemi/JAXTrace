"""Render the Marp meeting deck to PDF via pandoc + beamer.

    python3 make_pdf.py [deck.md] [out.pdf]

⚠️ Marp is not installed here, so this is NOT a pixel-faithful render of the Marp
deck -- it is the same content laid out by beamer. Differences to expect:
  * Marp's CSS classes (.box/.warn/.ok) become coloured LaTeX blocks;
  * image widths given as `![w:940]` become a fraction of the text width;
  * slide breaks (`---`) become frames.

The point is a shareable, printable PDF with every slide and figure present and in
order, not a replica of the HTML styling.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "damage_talk_meeting.md"
OUT = sys.argv[2] if len(sys.argv) > 2 else "damage_talk_meeting.pdf"
SLIDE_PX = 1160.0            # Marp usable width in px, for scaling image widths


def convert(md: str) -> str:
    # ⚠️ Drop the Marp front matter and the hidden-slide comment block first.
    if md.startswith("---"):
        end = md.index("\n---\n", 4)
        md = md[end + 5:]
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)

    # Marp image sizing -> a fraction of \textwidth that beamer understands.
    # ⚠️ Percentage widths, not \textwidth fractions: pandoc converts a bare
    # percentage itself, whereas a raw LaTeX length inside the attribute block is
    # passed through and breaks xelatex ("Illegal unit of measure").
    def img(m):
        w, path = int(m.group(1)), m.group(2)
        pct = int(min(w / SLIDE_PX, 1.0) * 100)
        return f"![]({path}){{width={pct}%}}"
    md = re.sub(r"!\[w:(\d+)\]\(([^)]+)\)", img, md)
    md = re.sub(r"!\[([^\]]*)\]\((figs_[^)]+)\)(?!\{)",
                lambda m: f"![]({m.group(2)}){{width=85%}}", md)

    # Callout divs -> beamer blocks, so the emphasis survives.
    # ⚠️ Fenced divs must balance. Count them and close any that the source left
    # open, otherwise pandoc swallows the rest of the deck into one block.
    # ⚠️ Match ANY div class, not just box|warn|ok. A `<div class="small">` on the
    # References slide was left as raw HTML, so pandoc passed its stray `</div>`
    # through to LaTeX and xelatex died on the unbalanced group.
    md = re.sub(r'<div class="(box|warn|ok)">\s*', r'\n::: {.block}\n', md)
    md = re.sub(r'<div class="(tiny|small)">\s*', r'\n::: {.\1}\n', md)
    md = re.sub(r'<div[^>]*>\s*', r'\n::: {.block}\n', md)
    md = md.replace("</div>", "\n:::\n")

    # Inline spans -> plain small text.
    # ⚠️ Do NOT wrap these in \footnotesize{...}: the captions contain braces and
    # LaTeX maths, so a brace wrapper is fragile and broke xelatex. Pandoc's own
    # fenced-div with a size class is handled safely by the beamer writer.
    md = re.sub(r'<span class="tiny">(.*?)</span>',
                lambda m: "\n::: {.footnotesize}\n" + m.group(1).strip() + "\n:::\n",
                md, flags=re.S)
    md = re.sub(r'<span class="small">(.*?)</span>',
                lambda m: "\n::: {.small}\n" + m.group(1).strip() + "\n:::\n",
                md, flags=re.S)
    md = re.sub(r"</?b>", "**", md)
    md = re.sub(r"</?i>", "*", md)
    md = re.sub(r"<br\s*/?>", "\n", md)
    md = re.sub(r"<sub>(.*?)</sub>", r"\1", md, flags=re.S)
    md = re.sub(r"<!-- _class: lead -->", "", md)

    # ⚠️ Balance the fenced divs LAST, after every span has become one.
    opens = len(re.findall(r"^::: \{\.", md, flags=re.M))
    closes = len(re.findall(r"^:::\s*$", md, flags=re.M))
    if closes < opens:
        md += "\n" + "\n:::\n" * (opens - closes)
    return md


def main() -> int:
    md = convert(open(SRC).read())
    tmp = "/tmp/_deck_for_pandoc.md"
    open(tmp, "w").write(md)

    cmd = [
        "pandoc", tmp, "-o", OUT,
        "-t", "beamer",
        "--pdf-engine=xelatex",
        "--slide-level=2",            # `##` starts a frame
        "-V", "aspectratio=169",
        "-V", "theme=default",
        "-V", "colorlinks=true",
        "-V", "fontsize=9pt",
        "-V", "geometry:margin=1cm",
        "--resource-path", ".",
    ]
    print("  " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        return 1
    size = os.path.getsize(OUT)
    print(f"\n  wrote {OUT}  ({size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
