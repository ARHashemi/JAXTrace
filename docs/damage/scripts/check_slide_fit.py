"""Geometric fit check for the Marp deck — does every slide fit a 16:9 frame?

    python3 check_slide_fit.py [deck.md]

⚠️ Why not just count lines. A line of body text and a 7-panel figure consume very
different vertical space, and `![w:1000]` sets the image WIDTH — its height then
follows from the file's own aspect ratio, which the Markdown does not state. A deck
can pass a line-count check and still overflow because one figure is 1.3x taller
than it is wide.

This measures instead:
  * image height  = (declared width px) x (actual file aspect) converted to slide px
  * text height   = lines x line-height, at the size the stylesheet gives that element
and compares the total against the usable frame.

Marp default 16:9 is 1280x720 px with the padding set in the deck's own style block.
"""
from __future__ import annotations

import os
import re
import sys

SLIDE_W, SLIDE_H = 1280, 720
PAD_TOP, PAD_BOTTOM = 50, 50            # from `section { padding: 50px 60px }`
PAD_X = 60
USABLE_H = SLIDE_H - PAD_TOP - PAD_BOTTOM
USABLE_W = SLIDE_W - 2 * PAD_X

# Line heights in slide px, from the deck's style block.
H_BODY = 25 * 1.45
H_H1 = 44 * 1.3
H_H2 = 34 * 1.3
H_TINY = 17 * 1.4
H_SMALL = 20 * 1.4
H_BOX = 21 * 1.38                        # .box / .warn / .ok
H_TABLE_ROW = 21 * 1.75                  # table rows carry padding
CHARS_PER_LINE = 92                      # ~ at 25px over a 1160px usable width


def img_height(path: str, declared_w: int) -> float:
    """Rendered height in slide px for `![w:N](path)`."""
    try:
        from PIL import Image
        w, h = Image.open(path).size
    except Exception:
        return float("nan")
    return declared_w * (h / w)


def measure(slide: str, base: str) -> tuple[float, list[str]]:
    total, notes = 0.0, []
    in_box = False
    for raw in slide.strip().split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("<!--") or line.startswith("<sub"):
            continue
        m = re.match(r"!\[w:(\d+)\]\(([^)]+)\)", line)
        if m:
            w, p = int(m.group(1)), os.path.join(base, m.group(2))
            h = img_height(p, w)
            total += h + 16
            notes.append(f"image {os.path.basename(p)} -> {h:.0f}px")
            continue
        if re.match(r'<div class="(box|warn|ok)"', line):
            in_box = True
            total += 20
            continue
        if line == "</div>":
            in_box = False
            total += 20
            continue
        if line.startswith("# "):
            total += H_H1
            continue
        if line.startswith("## "):
            total += H_H2 + 12
            continue
        if line.startswith("|"):
            total += H_TABLE_ROW
            continue
        if 'class="tiny"' in line:
            txt = re.sub(r"<[^>]+>", "", line)
            total += H_TINY * max(1, len(txt) // 130 + 1)
            continue
        if 'class="small"' in line:
            total += H_SMALL
            continue
        txt = re.sub(r"<[^>]+>", "", line)
        h = H_BOX if in_box else H_BODY
        total += h * max(1, len(txt) // CHARS_PER_LINE + 1)
    return total, notes


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "damage_implementation_talk.md"
    base = os.path.dirname(os.path.abspath(path)) or "."
    s = open(path).read()
    # ⚠️ Strip HTML comment blocks first: a deck may park hidden slides inside
    # <!-- ... -->, and those must not be measured as if they were shown.
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    body = s.split("---\n", 2)[2] if s.startswith("---") else s
    slides = body.split("\n---\n")

    bad = []
    for n, sl in enumerate(slides, 1):
        h, notes = measure(sl, base)
        title = next((l for l in sl.strip().split("\n") if l.startswith("#")), "(none)")
        pct = 100 * h / USABLE_H
        if h > USABLE_H:
            bad.append((n, h, pct, title[:58], notes))
    print(f"  {len(slides)} slides, usable height {USABLE_H}px\n")
    if not bad:
        print("  ✅ every slide fits")
        return 0
    print(f"  ⚠️ {len(bad)} slide(s) OVERFLOW:\n")
    for n, h, pct, title, notes in bad:
        print(f"   slide {n:3d}  {h:6.0f}px  ({pct:5.1f}% of frame)  {title}")
        for x in notes:
            print(f"             {x}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
