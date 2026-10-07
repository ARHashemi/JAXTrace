"""Render FUTURE_MODELS_EXPLAINED.md to a PDF for reading away from the terminal.

    python3 make_doc_pdf.py [input.md] [-o output.pdf]

⚠️ Why this is separate from make_pdf.py. That script targets `-t beamer` and treats
every `## ` as a new slide. This document is prose with deep section nesting, so
beamer would shatter it into ~90 fragments. Here the target is an article with a
table of contents and numbered sections.

⚠️ The ranked comparison table in §6 is 7 columns wide. LaTeX's default tabular
does not wrap cells, so that table would run off the page; `-t latex` with
pandoc's default longtable handling sizes columns to the text width and wraps
inside them, which is why no custom table code is needed here.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

DEFAULT_IN = "FUTURE_MODELS_EXPLAINED.md"


def main() -> int:
    src = DEFAULT_IN
    out = None
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "-o" and i + 1 < len(args):
            out = args[i + 1]
        elif not a.startswith("-") and (i == 0 or args[i - 1] != "-o"):
            src = a
    if out is None:
        out = os.path.splitext(src)[0] + ".pdf"
    if not os.path.exists(src):
        print(f"  no such file: {src}")
        return 1

    # ⚠️ Latin Modern has no U+26A0, and xelatex DROPS a missing glyph while only
    # WARNING -- so the caveat marks would vanish from the PDF with no error.
    # mainfontfallback is not honoured by this pandoc/xelatex pair, so the two
    # characters are rewritten to text that always renders. U+FE0F is an invisible
    # emoji variation selector and is simply removed.
    md = open(src, encoding="utf-8").read()
    md = md.replace("\u26a0\ufe0f", "\u26a0").replace("\ufe0f", "")
    md = md.replace("\u26a0", r"**NOTE**")

    # ⚠️ Unicode SUPERSCRIPTS are the dangerous case: Latin Modern lacks most of
    # them, so "8.03x10\u00b2\u2076 s\u207b\u00b9" silently printed as "8.03x10 s" -- a rate
    # constant turned into a different number with no error raised. Each run is
    # rewritten to real LaTeX maths so the exponent is typeset, not dropped.
    SUP = {"\u2070": "0", "\u00b9": "1", "\u00b2": "2", "\u00b3": "3", "\u2074": "4",
           "\u2075": "5", "\u2076": "6", "\u2077": "7", "\u2078": "8", "\u2079": "9",
           "\u207b": "-", "\u207a": "+"}
    SUB = {"\u2080": "0", "\u2081": "1", "\u2082": "2", "\u2083": "3", "\u2084": "4",
           "\u2085": "5", "\u2086": "6", "\u2087": "7", "\u2088": "8", "\u2089": "9"}

    def _runs(text, table, wrap):
        chars = "".join(table)
        return re.sub("[" + chars + "]+",
                      lambda m: wrap % "".join(table[c] for c in m.group(0)),
                      text)

    md = _runs(md, SUP, "$^{%s}$")
    md = _runs(md, SUB, "$_{%s}$")

    # ⚠️ The tick/cross marks CARRY MEANING in the ranked comparison table (§6):
    # a dropped glyph leaves the cell blank, which reads as "no information"
    # rather than "yes" or "no". Mapped to LaTeX symbols that always render.
    GLYPH = {
        "\u2705": r"$\checkmark$", "\u2714": r"$\checkmark$",
        "\u274c": r"$\times$", "\u2717": r"$\times$", "\u2718": r"$\times$",
        "\u26a1": r"**!**", "\u2b50": r"$\star$", "\U0001f7e1": r"(partial)",
        "\U0001f7e2": r"(good)", "\U0001f534": r"(poor)", "\U0001f7e0": r"(mixed)",
        "\u2192": r"$\rightarrow$", "\u2190": r"$\leftarrow$",
        # Maths relations outside Latin Modern's text coverage. These change the
        # MEANING of a sentence if dropped ("A \u221d B" -> "A B"), so none may be
        # left to the silent-drop path.
        "\u221d": r"$\propto$", "\u2248": r"$\approx$", "\u2260": r"$\neq$",
        "\u2264": r"$\leq$", "\u2265": r"$\geq$", "\u00d7": r"$\times$",
        "\u2212": r"$-$", "\u00b7": r"$\cdot$", "\u2202": r"$\partial$",
        "\u221a": r"$\sqrt{\ }$", "\u221e": r"$\infty$", "\u2211": r"$\sum$",
        "\u222b": r"$\int$", "\u00b1": r"$\pm$",
        "\u2237": r"$\therefore$",
    }
    # ⚠️ Inside bold (`**\u22482**`) a `$\approx$` is typeset by the bold TEXT font,
    # lmroman10-bold, which has no \u2248 -- so it was still dropped. \ensuremath
    # forces maths mode irrespective of the surrounding \textbf, which renders in
    # every context. Pandoc passes the raw macro through to LaTeX untouched.
    for k, v in GLYPH.items():
        if v.startswith("$") and v.endswith("$") and len(v) > 2:
            v = r"\ensuremath{" + v[1:-1] + "}"
        md = md.replace(k, v)
    tmp = os.path.join(os.path.dirname(os.path.abspath(out)) or ".",
                       "_doc_for_pandoc.md")
    open(tmp, "w", encoding="utf-8").write(md)
    src = tmp

    # ⚠️ The document's single `# ` line is its TITLE, not a section. Left as a
    # section it nests everything under "1", so §6 renders as "1.13". Pandoc only
    # promotes it when it is passed as metadata, hence --shift-heading-level-by=-1
    # (the `#` becomes the title and `##` becomes a top-level numbered section).
    cmd = [
        "pandoc", src, "-o", out,
        "-t", "latex", "--pdf-engine=xelatex",
        "--shift-heading-level-by=-1",
        "--toc", "--toc-depth=3", "--number-sections",
        "-V", "documentclass=article",
        "-V", "geometry:a4paper,margin=2.2cm",
        "-V", "fontsize=10pt",
        "-V", "colorlinks=true",
        "-V", "linkcolor=blue",
        # ⚠️ Long equations and code spans must not run into the margin.
        "-V", "monofontoptions=Scale=0.78",
        "--highlight-style=tango",
        "--resource-path", ".",
    ]
    print("  " + " ".join(cmd) + "\n")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        return r.returncode
    if r.stderr.strip():
        print(r.stderr.strip()[:1500] + "\n")
    mb = os.path.getsize(out) / 1e6
    print(f"  wrote {out}  ({mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
