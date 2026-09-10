"""Does the left-preference found in study_seam.csv contaminate the human
numbers already published?

Round 67 killed the seam study's main criterion and, worse, found the
instrument was at fault: 39/55 = 71% of picks went to the left card
(p=0.0027), median 2.9s per pair. Every human study in this repo was built
from the same template and shows each pair exactly ONCE -- so the same
failure mode could be sitting inside the numbers that are already in
main.tex (Sec. 4.1's 60.3%, Sec. 4.2, B2).

Why raw left-rate is a clean measure of position bias
-----------------------------------------------------
Side assignment is balanced by construction (build_study*.py alternates via
a counter). If arm A sits on the left in n_AL pairs and on the right in n_AR
pairs, then under ANY content preference whatsoever, P(pick left) = 1/2 as
long as n_AL == n_AR. So a raw left-rate away from 50% is evidence about the
annotator's side preference, NOT about the content. That is what makes this
auditable after the fact, without re-running anything.

How much can it move a published number
---------------------------------------
Under a pure-position-bias null (annotator picks left with probability L,
ignoring content entirely):

    E[A wins] / n = (L*n_AL + (1-L)*n_AR) / n
    bias    = E[A wins]/n - 1/2 = (L - 1/2) * (n_AL - n_AR) / n

So the damage is the product of two things: how lopsided the annotator is,
and how lopsided the side assignment happened to be. Perfect side balance
cancels a position bias exactly (it costs precision, not correctness).

Pre-registered criteria (fixed before running; see git history)
---------------------------------------------------------------
  C1 "bias present"      : two-sided exact binomial on left-rate over real,
                           non-tie rows, p < 0.05.
  C2 "contamination material": |L - 1/2| * |n_AL - n_AR| / n >= 0.02, i.e.
                           position bias alone could move the headline by
                           >= 2 percentage points.
  C3 "shape check"       : report the win-rate for the focal arm split by
                           the side it appeared on. Round 67's seam data
                           showed 82% vs 41% -- the shape of a side
                           preference, not of a content effect. Diagnostic
                           only; does not decide anything.

Verdict per study: a published number is flagged ONLY if C1 and C2 both
hold. C1 alone means the instrument is noisy (effective n is below nominal
n) but the point estimate stays unbiased.

Exact binomial via math.comb -- the remote env has no scipy.
"""

import argparse
import csv
import math
from pathlib import Path

# Human-labelled CSVs in the repo, with the arm whose win-rate is published.
# focal=None means "just report both arms".
STUDIES = [
    ("annotations.csv", None),
    ("study_labels.csv", None),
    ("a4_labels.csv", "model"),
    ("b2_labels.csv", "model"),
    ("study_seam.csv", "seam"),
]


def binom_two_sided(k, n, p=0.5):
    """Exact two-sided binomial test by summing tails no more likely than k."""
    if n == 0:
        return 1.0
    obs = math.comb(n, k) * p ** k * (1 - p) ** (n - k)
    tot = 0.0
    for j in range(n + 1):
        pr = math.comb(n, j) * p ** j * (1 - p) ** (n - j)
        if pr <= obs * (1 + 1e-9):
            tot += pr
    return min(1.0, tot)


def audit(path, focal):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    real = [r for r in rows if r.get("kind") == "real"]
    check = [r for r in rows if r.get("kind") == "check"]
    decided = [r for r in real if r.get("choice") in ("left", "right")]
    ties = len(real) - len(decided)

    n = len(decided)
    left = sum(1 for r in decided if r["choice"] == "left")
    L = left / n if n else 0.5
    p_bias = binom_two_sided(left, n)

    print("=" * 72)
    print(f"{path.name}   rows {len(rows)}   real {len(real)}   check {len(check)}"
          f"   ties(discarded) {ties}")

    if check:
        bad = sum(1 for r in check
                  if r.get("choice") != "tie" and r.get("chosen") != "good")
        print(f"  attention checks: {len(check) - bad}/{len(check)} correct")

    print(f"  C1 left-rate = {left}/{n} = {L:.1%}   p = {p_bias:.4f}"
          f"   -> bias {'PRESENT' if p_bias < 0.05 else 'not detected'}")

    # Which arms exist, and how the sides fell.
    arms = sorted({r["left"] for r in decided} | {r["right"] for r in decided})
    if focal is None and len(arms) == 2:
        focal = arms[0]

    worst = 0.0
    for arm in arms:
        n_al = sum(1 for r in decided if r["left"] == arm)
        n_ar = sum(1 for r in decided if r["right"] == arm)
        if n_al + n_ar != n:
            continue
        skew = abs(n_al - n_ar) / n
        contam = abs(L - 0.5) * skew
        worst = max(worst, contam)
        wins = sum(1 for r in decided if r["chosen"] == arm)
        # C3: split the arm's win-rate by the side it sat on.
        wl = sum(1 for r in decided if r["left"] == arm and r["choice"] == "left")
        wr = sum(1 for r in decided if r["right"] == arm and r["choice"] == "right")
        star = "  <-- published" if arm == focal else ""
        print(f"  arm {arm!r}: wins {wins}/{n} = {wins/n:.1%}"
              f"   p = {binom_two_sided(wins, n):.4f}{star}")
        print(f"      sides L/R = {n_al}/{n_ar}  (skew {skew:.1%})"
              f"   C2 max shift from position bias = {contam*100:.2f} pp")
        print(f"      C3 shape: won {wl}/{n_al} when on left"
              f" ({wl/n_al:.0%})" if n_al else "      C3 shape: n/a")
        if n_ar:
            print(f"                won {wr}/{n_ar} when on right ({wr/n_ar:.0%})")

    times = [int(r["ms"]) for r in real if r.get("ms", "").strip().isdigit()]
    if times:
        times.sort()
        med = times[len(times) // 2]
        fast = sum(1 for t in times if t < 2000)
        print(f"  timing: median {med/1000:.1f}s   under 2s: {fast}/{len(times)}")
    else:
        print("  timing: not recorded by this build of the template")

    flagged = p_bias < 0.05 and worst >= 0.02
    print(f"  VERDICT: {'FLAGGED (C1 and C2 both hold)' if flagged else 'clean'}"
          f"   [C1 p={p_bias:.4f}, C2 max={worst*100:.2f} pp]")
    return flagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("experiments/annotate"))
    a = ap.parse_args()

    flags = []
    for name, focal in STUDIES:
        path = a.dir / name
        if not path.exists():
            print(f"(missing: {name})")
            continue
        if audit(path, focal):
            flags.append(name)

    print("=" * 72)
    if flags:
        print("FLAGGED studies (published number may be shifted): " + ", ".join(flags))
    else:
        print("No published human number is shifted by position bias.")
        print("Side balance cancels it; the cost is precision, not correctness.")


if __name__ == "__main__":
    main()
