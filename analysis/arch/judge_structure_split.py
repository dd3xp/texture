"""Does TRD lose to B2 *specifically on materials with periodic structure*?

Pre-registration (written and committed BEFORE any outcome was inspected).
Zero GPU, zero API: reads only judge JSONs already in experiments/ plus the
artist tiles in data/tiles/dataset_k16.json.

Why ask.  The ledger's diagnosis is "TRD's structure has a weak relation to the
material name" (struct_sheet.png, eyeballed), and v7 bets on structure exemplars
because of it.  That diagnosis has never been measured.  The three judge runs on
the validation set all scored the *same* 125 materials, so the data to test it is
already on disk.

------------------------------------------------------------------ definitions
Structure score S(material).  For each 16x16 artist tile of that material in
split=val we take the luminance-rank grid and compute, over every non-trivial
cyclic shift (dy, dx),

    agree(dy, dx) = fraction of cells whose rank equals the rank at (y+dy, x+dx)

and subtract the chance level for that tile's own rank histogram,
chance = sum_c p_c^2 (the probability two cells drawn at random carry the same
rank).  S_tile = max_{(dy,dx) != (0,0)} agree - chance; S(material) = mean over
that material's tiles.  The subtraction is what makes a flat tile score 0
instead of 1: for a single-colour tile every shift agrees perfectly, but so does
chance.  High S = "this tile repeats on a short lattice" (brick courses, plank
rows); S ~ 0 = "cells are exchangeable noise" (sand, gravel).

Split.  Materials are cut at the **median** S into HI (structured) and LO.

--------------------------------------------------------------------- criteria
Arms, all scored against the same opponent B2 by the same judge under the same
two-order protocol: REAL (artist tiles), RR4 (TRD + CLIP-B/16 rerank), SINGLE
(TRD single sample).  verdict "A" = the named arm wins, "B" = B2 wins,
"inconsistent" = the two orders disagreed and the pair is discarded.

  (1) PRIMARY.  The artist-minus-TRD gap is larger on HI than on LO, measured on
      the materials decided in *both* arms (paired).  Reported for RR4 and for
      SINGLE.  This is the claim "the deficit is a structure deficit".
      Supported iff gap_HI - gap_LO > 0 and the 2x2 (arm x half) Fisher test on
      the TRD arm's own wins reaches p < 0.05.
  (2) SECONDARY.  Per-arm win rate within each half, with an exact two-sided
      binomial against 0.5.
  (3) DECIDABILITY.  Fraction of pairs the judge could resolve, HI vs LO.  If
      the judge simply cannot resolve structured materials, (1) measures the
      judge, not the model - so this has to be read before (1).

A null result is informative: if the deficit is flat across S, then structure
exemplars (v7) are aimed at the wrong thing and the loss is about something
else (contrast, palette, single-tile polish).

Selftest (--selftest): shuffling a tile's cells must drive S to ~0, a tiled 4x4
motif must score high, a flat tile must score 0, and the per-arm totals
recomputed from the records must reproduce the published 59/89, 42/90, 34/98.

------------------------------------------------- 2026-09-11: arms are now CLI
The split (S, median cut) is frozen; --arms points the same ruler at new judge
runs and --compare A,B adds a paired arm-vs-arm test inside each half.

Second pre-registration, written and committed before running it on v7.  The
totals (v7@20k+exemplars 34/99, v4+choice_temp20 26/98) were already known;
*where* that difference sits was not.  Question: do structure exemplars close
the deficit that the first run located entirely on the HI half?

  CT20  = v4 (clean pack data) + retrieval palette + choice_temp 20, 1 sample
  V7EX  = v7@20k (mod data + domain cond + 4 structure exemplars), 1 sample

  The two arms differ by more than the exemplars (training run and training
  data differ too); v8 = clean data + exemplars is what isolates the factor.
  Read the result as "does the v7 package help on structured materials".

  (3) DECIDABILITY, read first: every (arm, half) cell must resolve >= 60% of
      its pairs, otherwise that cell measures the judge, not the model.
  (1) PRIMARY: on HI materials decided in *both* arms, V7EX beats CT20 and the
      exact McNemar on the discordant HI pairs reaches p < 0.05.
  (2) SECONDARY: the same on LO, plus each arm's artist-minus-arm gap per half.

  A null on (1) together with a positive (2) would mean the v7 package helps on
  the half where nothing was missing -- that closes the exemplar route instead
  of extending it.
"""

import argparse
import collections
import json
import math
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET = os.path.join(ROOT, "data", "tiles", "dataset_k16.json")
ARMS = [
    ("REAL", "judge_full_REALval_vs_B2val_16_V_mat.json", (59, 89)),
    ("RR4", "judge_full_v2_retm4_rr4_vs_B2val_16_V_mat.json", (42, 90)),
    ("SINGLE", "judge_full_v2_retmM_c1.5_vs_B2val_16_V_mat.json", (34, 98)),
]


def binom_p(k, n):
    """Exact two-sided binomial test against p=0.5 (no scipy on the servers)."""
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) for i in range(n + 1)]
    total = float(2 ** n)
    obs = probs[k]
    return min(1.0, sum(p for p in probs if p <= obs + 1e-9) / total)


def fisher_2x2(a, b, c, d):
    """Two-sided Fisher exact test on [[a, b], [c, d]]."""
    n = a + b + c + d
    if n == 0:
        return 1.0
    r1, c1 = a + b, a + c

    def prob(x):
        return (
            math.comb(r1, x)
            * math.comb(n - r1, c1 - x)
            / math.comb(n, c1)
        )

    lo = max(0, c1 - (n - r1))
    hi = min(r1, c1)
    obs = prob(a)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= obs + 1e-12))


def structure_score(grid):
    """Excess periodic agreement of one luminance-rank grid (see module docstring).

    grid: side x side list/array of ints.  (Fixed 2026-09-11: this used to take the
    dataset's hex string and read it one character per cell, but the string is two hex
    characters per cell -- so it scored only the first half of each tile with a fake 0
    inserted in every other column, i.e. an artificial period-2 pattern.  The selftest
    fed the same wrong one-char-per-cell format, so it could not catch it.)"""
    side = len(grid)
    grid = [[int(v) for v in row] for row in grid]
    counts = collections.Counter(c for row in grid for c in row)
    n = float(side * side)
    chance = sum((v / n) ** 2 for v in counts.values())
    best = 0.0
    for dy in range(side):
        for dx in range(side):
            if dy == 0 and dx == 0:
                continue
            same = 0
            for y in range(side):
                row, srow = grid[y], grid[(y + dy) % side]
                for x in range(side):
                    if row[x] == srow[(x + dx) % side]:
                        same += 1
            best = max(best, same / n)
    return best - chance


def decode_idx(hexstr, side=16):
    """dataset_k16.json stores idx as bytes.hex(): two hex characters per cell."""
    vals = bytes.fromhex(hexstr)
    assert len(vals) == side * side, len(vals)
    return [[vals[y * side + x] for x in range(side)] for y in range(side)]


def load_scores(prompt_sets):
    data = json.load(open(DATASET, encoding="utf-8"))
    by_material = collections.defaultdict(list)
    for s in data["samples"]:
        if s["split"] == "val" and s["size"] == 16:
            by_material[s["material"]].append(s["idx"])
    scores = {}
    for entry in prompt_sets["V_mat"]:
        tiles = by_material.get(entry["material"], [])
        if not tiles:
            continue
        scores[entry["prompt"]] = statistics.mean(structure_score(decode_idx(t)) for t in tiles)
    return scores


def rate_line(name, wins, dec):
    if dec == 0:
        return "%-10s   n/a (no decided pairs)" % name
    return "%-10s %3d/%-3d = %4.1f%%   p=%.4f" % (
        name, wins, dec, 100.0 * wins / dec, binom_p(wins, dec),
    )


def selftest():
    side = 16
    flat = [[0] * side for _ in range(side)]
    assert abs(structure_score(flat)) < 1e-9, structure_score(flat)

    motif = [[(y // 2 + x // 4) % 3 for x in range(side)] for y in range(side)]
    s_tiled = structure_score(motif)
    assert s_tiled > 0.4, s_tiled
    # the real storage format must round-trip (this is the check that was missing)
    hexstr = bytes(v for row in motif for v in row).hex()
    assert decode_idx(hexstr) == motif
    assert abs(structure_score(decode_idx(hexstr)) - s_tiled) < 1e-12

    import random
    rng = random.Random(0)
    cells = [v for row in motif for v in row]
    rng.shuffle(cells)
    s_shuf = structure_score([cells[y * side:(y + 1) * side] for y in range(side)])
    assert s_shuf < 0.15, s_shuf
    assert s_tiled - s_shuf > 0.3, (s_tiled, s_shuf)

    # binomial / Fisher sanity
    assert abs(binom_p(5, 10) - 1.0) < 1e-9
    assert binom_p(59, 89) < 0.005, binom_p(59, 89)
    assert abs(fisher_2x2(10, 0, 0, 10) - 2 * math.comb(10, 10) * math.comb(10, 0)
               / math.comb(20, 10)) < 1e-9

    for name, fname, (w, d) in ARMS:
        path = os.path.join(ROOT, "experiments", fname)
        if not os.path.exists(path):
            print("selftest: SKIP totals for %s (file absent)" % name)
            continue
        recs = json.load(open(path, encoding="utf-8"))["records"]
        wins = sum(1 for r in recs if r["verdict"] == "A")
        dec = sum(1 for r in recs if r["verdict"] in ("A", "B"))
        assert (wins, dec) == (w, d), (name, wins, dec, w, d)
        print("selftest: %s totals reproduce %d/%d" % (name, w, d))
    print("selftest: structure score flat=0.000 tiled=%.3f shuffled=%.3f" % (s_tiled, s_shuf))
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--prompt-sets", default=os.path.join(ROOT, "eval", "prompt_sets_val.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments", "judge_structure_split.json"))
    ap.add_argument("--arms", default=None,
                    help="comma-separated NAME=judge_full_file.json; replaces the "
                         "non-REAL arms (REAL is always kept, it defines the split)")
    ap.add_argument("--compare", default=None,
                    help="NAME_A,NAME_B: paired McNemar of A against B inside each half")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return 0

    prompt_sets = json.load(open(args.prompt_sets, encoding="utf-8"))
    scores = load_scores(prompt_sets)

    arm_files = [(n, f) for n, f, _ in ARMS]
    if args.arms:
        arm_files = [(n, f) for n, f, _ in ARMS if n == "REAL"]
        for spec in args.arms.split(","):
            name, _, fname = spec.partition("=")
            assert fname, "expected NAME=file.json, got %r" % spec
            arm_files.append((name, fname))

    arms = {}
    for name, fname in arm_files:
        path = os.path.join(ROOT, "experiments", fname)
        if not os.path.exists(path):
            print("missing judge file: %s" % path)
            return 2
        recs = json.load(open(path, encoding="utf-8"))["records"]
        arms[name] = {r["material"]: r["verdict"] for r in recs}

    materials = sorted(set(scores) & set(arms["REAL"]))
    cut = statistics.median(scores[m] for m in materials)
    half = {m: ("HI" if scores[m] >= cut else "LO") for m in materials}
    print("materials scored: %d   median structure score (cut): %.3f" % (len(materials), cut))
    print("HI n=%d  LO n=%d" % (
        sum(1 for m in materials if half[m] == "HI"),
        sum(1 for m in materials if half[m] == "LO"),
    ))

    print("\n(3) DECIDABILITY -- read this first")
    for name in arms:
        for h in ("HI", "LO"):
            ms = [m for m in materials if half[m] == h]
            dec = sum(1 for m in ms if arms[name][m] in ("A", "B"))
            rate = dec / len(ms)
            print("  %-7s %s  %3d/%-3d = %4.1f%% resolvable   %s"
                  % (name, h, dec, len(ms), 100.0 * rate,
                     "ok" if rate >= 0.60 else "BELOW 60% -- cell not interpretable"))

    print("\n(2) WIN RATE vs B2, per half")
    per_half = {}
    for name in arms:
        for h in ("HI", "LO"):
            ms = [m for m in materials if half[m] == h]
            wins = sum(1 for m in ms if arms[name][m] == "A")
            dec = sum(1 for m in ms if arms[name][m] in ("A", "B"))
            per_half[(name, h)] = (wins, dec)
            print("  " + rate_line("%s %s" % (name, h), wins, dec))

    print("\n(1) PRIMARY -- artist-minus-TRD gap, paired materials")
    primary = {}
    for name in [n for n, _ in arm_files if n != "REAL"]:
        gaps = {}
        for h in ("HI", "LO"):
            ms = [m for m in materials if half[m] == h
                  and arms["REAL"][m] in ("A", "B") and arms[name][m] in ("A", "B")]
            real = sum(1 for m in ms if arms["REAL"][m] == "A")
            trd = sum(1 for m in ms if arms[name][m] == "A")
            gaps[h] = {
                "n": len(ms),
                "real_wins": real,
                "trd_wins": trd,
                "real_rate": real / len(ms) if ms else None,
                "trd_rate": trd / len(ms) if ms else None,
            }
            print("  %-7s %s  n=%2d  artist %4.1f%%  TRD %4.1f%%  gap %+5.1fpp"
                  % (name, h, len(ms),
                     100.0 * real / len(ms), 100.0 * trd / len(ms),
                     100.0 * (real - trd) / len(ms)))
        hi, lo = gaps["HI"], gaps["LO"]
        gap_diff = (hi["real_rate"] - hi["trd_rate"]) - (lo["real_rate"] - lo["trd_rate"])
        a, b = per_half[(name, "HI")]
        c, d = per_half[(name, "LO")]
        p_fisher = fisher_2x2(a, b - a, c, d - c)
        ok = gap_diff > 0 and p_fisher < 0.05
        print("    gap_HI - gap_LO = %+5.1fpp;  TRD wins HI %d/%d vs LO %d/%d, Fisher p=%.4f  -> %s"
              % (100.0 * gap_diff, a, b, c, d, p_fisher,
                 "SUPPORTED" if ok else "not supported"))
        # POST-HOC (added after seeing (1) and (2); not part of the pre-registration).
        # The pre-registered Fisher test asks whether *TRD's own* win rate moves
        # between the halves. It does not. The gap moves because the *artist*
        # rate moves, so the paired artist-vs-TRD comparison on the same
        # materials is the test that matches what the numbers actually show.
        mcnemar = {}
        for h in ("HI", "LO"):
            ms = [m for m in materials if half[m] == h
                  and arms["REAL"][m] in ("A", "B") and arms[name][m] in ("A", "B")]
            b = sum(1 for m in ms if arms["REAL"][m] == "A" and arms[name][m] == "B")
            c = sum(1 for m in ms if arms["REAL"][m] == "B" and arms[name][m] == "A")
            mcnemar[h] = {"artist_only": b, "trd_only": c, "p": binom_p(b, b + c)}
            print("    post-hoc %s McNemar: artist-only %d, TRD-only %d, p=%.4f"
                  % (h, b, c, binom_p(b, b + c)))
        primary[name] = {
            "halves": gaps, "gap_diff": gap_diff, "fisher_p": p_fisher,
            "trd_hi": [a, b], "trd_lo": [c, d], "supported": ok,
            "posthoc_mcnemar": mcnemar,
        }

    compare = None
    if args.compare:
        na, nb = args.compare.split(",")
        print("\nPAIRED %s vs %s, materials decided in both arms" % (na, nb))
        compare = {}
        for h in ("HI", "LO"):
            ms = [m for m in materials if half[m] == h
                  and arms[na][m] in ("A", "B") and arms[nb][m] in ("A", "B")]
            wa = sum(1 for m in ms if arms[na][m] == "A")
            wb = sum(1 for m in ms if arms[nb][m] == "A")
            only_a = sum(1 for m in ms if arms[na][m] == "A" and arms[nb][m] == "B")
            only_b = sum(1 for m in ms if arms[na][m] == "B" and arms[nb][m] == "A")
            p = binom_p(only_a, only_a + only_b)
            compare[h] = {"n": len(ms), "wins_a": wa, "wins_b": wb,
                          "only_a": only_a, "only_b": only_b, "mcnemar_p": p}
            print("  %s  n=%2d  %s %4.1f%%  %s %4.1f%%  diff %+5.1fpp   "
                  "discordant %d:%d  McNemar p=%.4f"
                  % (h, len(ms), na, 100.0 * wa / len(ms) if ms else 0.0,
                     nb, 100.0 * wb / len(ms) if ms else 0.0,
                     100.0 * (wa - wb) / len(ms) if ms else 0.0,
                     only_a, only_b, p))
        hi = compare["HI"]
        ok = hi["wins_a"] > hi["wins_b"] and hi["mcnemar_p"] < 0.05
        compare["primary_supported"] = ok
        print("  (1) PRIMARY on HI -> %s" % ("SUPPORTED" if ok else "not supported"))

    out = {
        "cut": cut,
        "arms_used": dict(arm_files),
        "compare": compare,
        "n_materials": len(materials),
        "scores": {m: scores[m] for m in materials},
        "half": half,
        "verdicts": {name: {m: arms[name][m] for m in materials} for name in arms},
        "per_half": {"%s_%s" % k: v for k, v in per_half.items()},
        "primary": primary,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
