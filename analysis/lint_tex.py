"""Mechanical text lint for paper/main.tex: doubled words, straight quotes,
bad spacing before \\ref/\\cite. Reports only; fix by hand."""
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'paper/main.tex'
lines = open(path, encoding='utf-8').read().split('\n')
issues = []
for i, ln in enumerate(lines, 1):
    s = re.sub(r'(?<!\\)%.*', '', ln)  # strip comments
    for m in re.finditer(r'\b([A-Za-z]{2,})\s+\1\b', s, re.I):
        issues.append((i, 'DOUBLE', m.group(0), ln.strip()[:90]))
    if '"' in s:
        issues.append((i, 'QUOTE', '"', ln.strip()[:90]))
    for m in re.finditer(r'[A-Za-z] (\\ref|\\eqref|\\autoref)\b', s):
        issues.append((i, 'REFSPACE', m.group(0), ln.strip()[:90]))
    # hyphen where number range should use en-dash (digit-digit in prose)
    for m in re.finditer(r'(?<![\d.\-])(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)(?![\d.\-])', s):
        issues.append((i, 'RANGE', m.group(0), ln.strip()[:90]))
for i, kind, frag, ctx in issues:
    print(f'{i:5d} {kind:9s} {frag!r}  | {ctx}')

# ---- pass 2: cross-line doubles, a/an, punctuation ----
raw = open(path, encoding='utf-8').read()
# strip comments, join wrapped lines but remember approximate line numbers
lines = [re.sub(r'(?<!\\)%.*', '', ln) for ln in raw.split('\n')]
issues = []
# cross-line doubled words: last word of line == first word of next
for i in range(len(lines) - 1):
    m1 = re.search(r'\b([A-Za-z]{2,})\s*$', lines[i])
    m2 = re.match(r'\s*([A-Za-z]{2,})\b', lines[i + 1])
    if m1 and m2 and m1.group(1).lower() == m2.group(1).lower():
        issues.append((i + 1, 'XDOUBLE', m1.group(1), lines[i].strip()[-60:]))
# a/an agreement (heuristic; skip math and commands)
vowel_sound = re.compile(r'^(?:[aeiou]|hour|honest|honor|heir|[A-Z]\b)', re.I)
an_ok_consonant = re.compile(r'^(?:uni|use|user|usa|one|euro|ubiq)', re.I)
for i, s in enumerate(lines, 1):
    s2 = re.sub(r'\$[^$]*\$', '', s)
    s2 = re.sub(r'\\[A-Za-z]+(\{[^{}]*\})*', '', s2)
    for m in re.finditer(r'\b(a|an) ([A-Za-z][A-Za-z\-]*)', s2):
        art, word = m.groups()
        v = bool(vowel_sound.match(word)) and not an_ok_consonant.match(word)
        if art.lower() == 'a' and v:
            issues.append((i, 'A->AN', m.group(0), s.strip()[:90]))
        if art.lower() == 'an' and not v:
            issues.append((i, 'AN->A', m.group(0), s.strip()[:90]))
for i, s in enumerate(lines, 1):
    for m in re.finditer(r'[.,;:]{2,}|\s[.,;]', s):
        if m.group(0) not in ('..', ',,') and re.match(r'\s[.,;]', m.group(0)) and '\\' in s[:m.start()+2][-20:]:
            continue
        issues.append((i, 'PUNCT', m.group(0), s.strip()[:90]))
for i, kind, frag, ctx in issues:
    print(f'{i:5d} {kind:9s} {frag!r}  | {ctx}')
print('total', len(issues))
