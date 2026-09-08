"""Extract the vocabulary of paper/main.tex for a manual spelling audit.

Strips LaTeX commands, math, comments, and file paths, then prints every
unique alphabetic word (lowercased) with its count, sorted by count then
alphabetically. Words inside \\cite/\\ref/\\label arguments are dropped.
"""
import re
import sys
from collections import Counter
from pathlib import Path

TEX = Path(__file__).resolve().parent.parent / "paper" / "main.tex"
src = TEX.read_text(encoding="utf-8")

# Drop comments.
src = re.sub(r"(?<!\\)%.*", "", src)
# Drop math ($...$, \(...\), equation-ish environments).
src = re.sub(r"\$[^$]*\$", " ", src)
# Drop arguments of reference-like and path-like commands.
src = re.sub(
    r"\\(?:cite[tp]?|ref|label|eqref|includegraphics|input|bibliography"
    r"|bibliographystyle|usepackage|documentclass|url|href|texttt)"
    r"\*?(?:\[[^\]]*\])?\{[^}]*\}",
    " ",
    src,
)
# Drop remaining command names but keep their text arguments.
src = re.sub(r"\\[a-zA-Z@]+\*?", " ", src)

words = re.findall(r"[A-Za-z]+(?:['\u2019-][A-Za-z]+)*", src)
counts = Counter(w.lower() for w in words if len(w) > 1)

for w, c in sorted(counts.items(), key=lambda kv: (kv[1], kv[0])):
    print(f"{c:4d}  {w}")
print(f"-- {len(counts)} unique words, {sum(counts.values())} tokens", file=sys.stderr)
