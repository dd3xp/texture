"""Rebuild a blind-comparison page so each pair is asked in BOTH orders.

Why
---
Round 67's seam study failed its main criterion, and the post-mortem found
the instrument was the weak link: every pair was shown exactly once, 71% of
picks went to the left card, and the attention checks (sharp vs heavily
blurred) cannot catch careless labelling because they are correct even when
answered in 0.9s without comparing anything.

Every VLM judge in this project has always been asked twice, in both orders,
with disagreements discarded. The human instrument -- the one introduced
*because* judge discrimination had run out -- was weaker than the judges on
exactly the axis that mattered. `study_crop.html` (39 pairs) and
`study_ab60.html` (60 pairs) are still unlabelled and carry the same defect.
This script fixes them before they are annotated, not after.

No pixels are regenerated
-------------------------
Both pages were key-audited in round 30 (ab60 60/60, crop 39/39 byte-exact).
Regenerating would need a GPU and would invalidate that audit -- and
/mnt/data is full. So this script only re-arranges the items that are
already in the page: every image in the output is copied verbatim from the
input, and `verify()` asserts exactly that. The key audit still holds.

What changes
------------
1. Each real pair appears twice, sharing a `pair` id: once as built, once
   with left/right swapped.
2. The two presentations sit in separate blocks (all pairs once, then all
   pairs again in an independent order), so the second showing is never
   adjacent to the first and cannot be answered from short-term memory.
3. A second kind of attention check, `check_same`, is added: both cards are
   the SAME image and the correct answer is "cannot tell". Unlike the blur
   check, this one cannot be answered without actually comparing the two
   cards -- which is the failure mode the blur check missed. The original
   blur checks are kept (they still verify the annotator is looking).
4. The template now blocks inputs faster than 700ms and records a `pair`
   column.

Consistency is enforced at analysis time by `reduce_paired_orders.py`, which
discards any pair whose two answers disagree. That is the same rule the VLM
judge line has always used.

Usage:
  python analysis/annotate/build_paired_orders.py experiments/annotate/study_crop.html
  python analysis/annotate/build_paired_orders.py --selftest
"""

import argparse
import base64
import hashlib
import io
import json
import random
import re
import sys
from pathlib import Path

ITEMS_RE = re.compile(r"const ITEMS = (\[.*?\]);", re.S)


def load_items(path: Path):
    m = ITEMS_RE.search(path.read_text(encoding="utf-8"))
    if not m:
        raise SystemExit(f"no ITEMS array found in {path}")
    return json.loads(m.group(1))


def swapped(it):
    out = dict(it)
    out["left"], out["right"] = it["right"], it["left"]
    out["limg"], out["rimg"] = it["rimg"], it["limg"]
    return out


def build(items, n_same, seed):
    """Return the re-arranged item list."""
    rng = random.Random(seed)
    real = [it for it in items if it["kind"] == "real"]
    checks = [it for it in items if it["kind"] != "real"]

    for j, it in enumerate(real):
        it["pair"] = f"p{j:03d}"

    # A check whose correct answer is "cannot tell": the same image twice.
    same = []
    for k, it in enumerate(rng.sample(real, min(n_same, len(real)))):
        same.append({"material": it["material"], "label": it["label"],
                     "kind": "check_same", "struct": 0.0,
                     "stratum": it.get("stratum", ""),
                     "left": "same", "right": "same",
                     "limg": it["limg"], "rimg": it["limg"],
                     "pair": f"s{k:03d}"})

    # Two blocks. Every pair is asked once in block 1 and once in block 2.
    # Shuffling the two blocks independently is NOT enough: a pair can land
    # last in block 1 and first in block 2, i.e. back to back (the selftest
    # caught exactly this, min gap 1). So block 2 is drawn under a constraint.
    #
    # With real-rank i in block 1 and j in block 2, the final distance is at
    # least n + j - i. Requiring j >= i - slack with slack = n//2 therefore
    # guarantees a gap of at least ceil(n/2) presentations, while still
    # leaving the order random.
    n = len(real)
    first = [dict(it) for it in real]
    rng.shuffle(first)
    slack = n // 2
    rank1 = {it["pair"]: i for i, it in enumerate(first)}

    remaining = sorted(real, key=lambda it: rank1[it["pair"]])
    second = []
    for j in range(n):
        eligible = [it for it in remaining if rank1[it["pair"]] <= j + slack]
        pick = rng.choice(eligible)
        remaining.remove(pick)
        second.append(swapped(pick))

    # Spread the checks over both blocks. They must be INSERTED, not shuffled
    # in: reordering the real items here would void the gap guarantee above.
    extras = [dict(c) for c in checks] + same
    rng.shuffle(extras)
    half = len(extras) // 2

    def interleave(block, add):
        out = list(block)
        for c in add:
            out.insert(rng.randint(0, len(out)), c)
        return out

    return interleave(first, extras[:half]) + interleave(second, extras[half:])


def verify(src, out, n_same):
    """Every image must come from the source page, and the design must hold."""
    def h(b):
        return hashlib.sha256(b.encode()).hexdigest()

    src_imgs = {h(it[k]) for it in src for k in ("limg", "rimg")}
    for it in out:
        for k in ("limg", "rimg"):
            assert h(it[k]) in src_imgs, "output contains an image not in the source"

    real_src = [it for it in src if it["kind"] == "real"]
    real_out = [it for it in out if it["kind"] == "real"]
    assert len(real_out) == 2 * len(real_src), "each real pair must appear twice"

    by_pair = {}
    for j, it in enumerate(real_out):
        by_pair.setdefault(it["pair"], []).append((j, it))
    assert len(by_pair) == len(real_src), "pair ids must be one per source pair"

    gaps = []
    for pid, got in by_pair.items():
        assert len(got) == 2, f"{pid} appears {len(got)} times"
        (i1, a), (i2, b) = got
        assert {a["left"], a["right"]} == {b["left"], b["right"]}, f"{pid} arms differ"
        assert a["left"] != b["left"], f"{pid} shown in the same order twice"
        assert h(a["limg"]) == h(b["rimg"]) and h(a["rimg"]) == h(b["limg"]), \
            f"{pid} is not a clean left/right swap"
        gaps.append(abs(i2 - i1))

    # The construction guarantees at least ceil(n/2) presentations between the
    # two showings of a pair; check it rather than trusting it.
    need = -(-len(real_src) // 2)
    assert min(gaps) >= need, f"min gap {min(gaps)} < guaranteed {need}"

    n_same_out = sum(1 for it in out if it["kind"] == "check_same")
    assert n_same_out == min(n_same, len(real_src)), "wrong number of same-image checks"
    for it in out:
        if it["kind"] == "check_same":
            assert h(it["limg"]) == h(it["rimg"]), "same-image check is not identical"

    # Side balance, counted over all presentations of each arm.
    arms = sorted({it["left"] for it in real_out})
    bal = {a: (sum(1 for it in real_out if it["left"] == a),
               sum(1 for it in real_out if it["right"] == a)) for a in arms}
    for a, (l, r) in bal.items():
        assert l == r, f"arm {a} is not side-balanced: {l}/{r}"
    return min(gaps), bal


def emit(out_items, dest: Path):
    tpl = (Path(__file__).parent / "task_template.html").read_text(encoding="utf-8")
    assert "__ITEMS__" in tpl, "template lost its __ITEMS__ placeholder"
    assert "MIN_MS" in tpl and "r.pair" in tpl, \
        "template lacks the time gate or the pair column; refusing to build"
    html = tpl.replace("__ITEMS__", json.dumps(out_items, ensure_ascii=False))
    assert "__ITEMS__" not in html, "placeholder substitution failed"
    dest.write_text(html, encoding="utf-8")


def selftest():
    """Synthetic page: the design guarantees must hold, and be checkable."""
    def img(c):
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), c).save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode()

    src = []
    for j in range(12):
        a, b = img((j * 7 % 255, 0, 0)), img((0, j * 11 % 255, 0))
        first = j % 2 == 0
        src.append({"material": f"m{j}", "label": f"m{j}", "kind": "real",
                    "struct": 1.0, "stratum": "cropped",
                    "left": "before" if first else "after",
                    "right": "after" if first else "before",
                    "limg": a if first else b, "rimg": b if first else a})
    src.append({"material": "m0", "label": "m0", "kind": "check", "struct": 0.0,
                "stratum": "cropped", "left": "good", "right": "blur",
                "limg": img((1, 1, 1)), "rimg": img((2, 2, 2))})

    out = build([dict(s) for s in src], n_same=3, seed=0)
    gap, bal = verify(src, out, 3)
    print(f"[selftest] {len(src)} -> {len(out)} items, min gap {gap}, balance {bal}")
    assert gap >= 6, f"the two showings are too close together: gap {gap}"

    # Negative test: the verifier must REJECT a page that is not swapped.
    bad = [dict(it) for it in out]
    for it in bad:
        if it["kind"] == "real":
            it["left"], it["right"] = "before", "after"
    try:
        verify(src, bad, 3)
    except AssertionError:
        print("[selftest] negative test OK: unswapped page is rejected")
    else:
        print("[selftest] FAIL: verifier accepted an unswapped page")
        return 1

    # Negative test: a foreign image must be rejected.
    bad2 = [dict(it) for it in out]
    bad2[0] = dict(bad2[0])
    bad2[0]["limg"] = img((123, 45, 67))
    try:
        verify(src, bad2, 3)
    except AssertionError:
        print("[selftest] negative test OK: foreign image is rejected")
    else:
        print("[selftest] FAIL: verifier accepted a foreign image")
        return 1

    print("[selftest] all checks passed")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n-same", type=int, default=4)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        raise SystemExit(selftest())
    if not a.src:
        raise SystemExit("usage: build_paired_orders.py <study.html> [--out ...]")

    src = load_items(a.src)
    out_items = build([dict(s) for s in src], a.n_same, a.seed)
    gap, bal = verify(src, out_items, a.n_same)

    dest = a.out or a.src.with_name(a.src.stem + "_v2.html")
    emit(out_items, dest)

    n_real = sum(1 for it in out_items if it["kind"] == "real")
    print(f"{a.src.name} -> {dest.name}")
    print(f"  presentations: {len(out_items)}  (real {n_real} = {n_real // 2} pairs x 2 orders)")
    print(f"  checks: blur {sum(1 for it in out_items if it['kind'] == 'check')}"
          f"  same-image {sum(1 for it in out_items if it['kind'] == 'check_same')}")
    print(f"  min gap between the two showings of a pair: {gap} items")
    print(f"  side balance per arm (left/right): {bal}")
    print(f"  every image verified to come from {a.src.name}; key audit still holds")
    print(f"  written {dest.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
