"""Collapse a both-orders CSV into a one-row-per-pair CSV, discarding the
pairs whose two answers disagree.

This is deliberately a *reduction*, not a new analysis. The judgement stays
with `analyze_pair_study.py`, whose criteria were pre-registered in 65 and
are not touched here:

    python analysis/annotate/reduce_paired_orders.py study_crop_v2.csv -o r.csv
    python analysis/annotate/analyze_pair_study.py crop r.csv

Keeping the two steps apart matters. If this script also decided the
outcome, the pre-registered criteria would effectively have been rewritten
alongside the instrument, and nobody could tell afterwards which change did
the work.

Rules, fixed before any data exists
-----------------------------------
Consistency  A pair is KEPT only if both presentations name the same arm.
             Both "cannot tell" counts as a kept tie. One tie plus one
             decisive answer counts as a disagreement and is discarded.
             This is the rule every VLM judge in this project has been held
             to; round 67 found the human instrument was not.

Blur checks  Passed through unchanged as kind=check rows, so that
             `analyze_pair_study.py` applies its own pre-registered
             threshold (wrong > 1 voids the session) with no change.

Same-image checks  New in the v2 instrument: both cards are the same image,
             so "cannot tell" is the only correct answer and it cannot be
             produced without comparing the two cards. Answering a side once
             or twice is ambiguous (an annotator may dislike ties), but
             doing it on most of them means no comparison took place.
             Gate: wrong > 2 of 4 voids the session, exit code 1.

Note on the side diagnostic downstream: a kept pair answered one side once
and the other side once, and reduction keeps whichever presentation came
first on the page. The left/right split printed by `analyze_pair_study.py`
is therefore a coin flip per pair no matter how the annotator behaved -- it
looks like a position-bias check but cannot detect one. The real position
statistics are printed HERE, over all 2N presentations, which is the caliber
the round-67 audit used.

Exit codes: 0 ok / 1 session voided / 2 nothing to reduce.
Self-test:  python analysis/annotate/reduce_paired_orders.py --selftest
"""

import argparse
import csv
import math
import sys
from pathlib import Path

FIELDS = ["idx", "material", "kind", "struct", "stratum",
          "left", "right", "choice", "chosen", "ms", "pair"]
MAX_SAME_WRONG = 2


def binom_two_sided(k, n, p=0.5):
    if n == 0:
        return 1.0
    obs = math.comb(n, k) * p ** k * (1 - p) ** (n - k)
    return min(1.0, sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j)
                        for j in range(n + 1)
                        if math.comb(n, j) * p ** j * (1 - p) ** (n - j)
                        <= obs * (1 + 1e-9)))


def reduce_rows(rows):
    """Return (kept_rows, stats). Pure function so the selftest can drive it."""
    real = [r for r in rows if r.get("kind") == "real"]
    blur = [r for r in rows if r.get("kind") == "check"]
    same = [r for r in rows if r.get("kind") == "check_same"]

    by_pair = {}
    for r in real:
        by_pair.setdefault(r.get("pair", ""), []).append(r)

    kept, dropped, malformed = [], [], []
    for pid, got in sorted(by_pair.items()):
        if len(got) != 2:
            malformed.append(pid)
            continue
        a, b = got
        ca, cb = (a.get("chosen") or "").strip(), (b.get("chosen") or "").strip()
        if ca != cb:
            dropped.append((pid, ca, cb))
            continue
        out = dict(a)
        out["pair"] = pid
        kept.append(out)

    n_pairs = len(by_pair)
    agreed = len(kept)
    stats = {
        "pairs": n_pairs,
        "agreed": agreed,
        "dropped": dropped,
        "malformed": malformed,
        "blur": blur,
        "same": same,
        "presentations": real,
    }
    return kept, stats


def report(stats):
    n, agreed = stats["pairs"], stats["agreed"]
    real = stats["presentations"]

    print(f"pairs {n}   presentations {len(real)}")
    if stats["malformed"]:
        print(f"  WARNING: {len(stats['malformed'])} pair id(s) without exactly "
              f"two presentations: {stats['malformed'][:5]}")

    rate = agreed / n if n else 0.0
    print(f"  order-swap agreement: {agreed}/{n} = {rate:.1%}"
          f"   (WebDevJudge judges: 83.5-89.6%)")
    print(f"  discarded as inconsistent: {len(stats['dropped'])}")

    blur, same = stats["blur"], stats["same"]
    bw = sum(1 for r in blur if (r.get("chosen") or "").strip() != "good")
    sw = sum(1 for r in same if (r.get("choice") or "").strip() != "tie")
    print(f"  blur checks wrong {bw}/{len(blur)}"
          f"   (threshold applied downstream by analyze_pair_study.py)")
    print(f"  same-image checks wrong {sw}/{len(same)}"
          f"   (gate here: wrong > {MAX_SAME_WRONG} voids the session)")

    decided = [r for r in real if (r.get("choice") or "").strip() in ("left", "right")]
    if decided:
        left = sum(1 for r in decided if r["choice"].strip() == "left")
        p = binom_two_sided(left, len(decided))
        print(f"  position: left {left}/{len(decided)} = {left/len(decided):.1%}"
              f"   p = {p:.4f}"
              f"   -> {'PRESENT' if p < 0.05 else 'not detected'}")
        print("     (with both orders asked, a side preference now shows up as"
              " disagreement and is discarded, instead of passing as signal)")

    times = sorted(int(r["ms"]) for r in real
                   if str(r.get("ms", "")).strip().isdigit())
    if times:
        print(f"  timing: median {times[len(times)//2]/1000:.1f}s"
              f"   under 2s: {sum(1 for t in times if t < 2000)}/{len(times)}")

    if sw > MAX_SAME_WRONG:
        print("  -> SESSION VOIDED: the same-image checks show the two cards"
              " were not being compared.")
        return 1
    if not blur and not same:
        print("  -> no check items found; refusing to reduce.")
        return 2
    return 0


def selftest():
    def mk(pid, kind, chosen, choice, left="after", right="before", ms=3000):
        return {"idx": "0", "material": "m", "kind": kind, "struct": "1",
                "stratum": "cropped", "left": left, "right": right,
                "choice": choice, "chosen": chosen, "ms": str(ms), "pair": pid}

    def pair(pid, c1, c2):
        a = mk(pid, "real", c1, "left" if c1 == "after" else "right",
               left="after", right="before")
        b = mk(pid, "real", c2, "left" if c2 == "before" else "right",
               left="before", right="after")
        return [a, b]

    bad = 0
    cases = []

    # 1. All consistent -> everything kept.
    rows = sum([pair(f"p{i}", "after", "after") for i in range(10)], [])
    rows += [mk("c0", "check", "good", "left"), mk("s0", "check_same", "tie", "tie")]
    kept, st = reduce_rows(rows)
    cases.append(("all consistent", len(kept) == 10 and st["agreed"] == 10, 0, report(st)))

    # 2. A pure side-picker: every pair disagrees -> everything discarded.
    rows = []
    for i in range(10):
        a = mk(f"p{i}", "real", "after", "left", left="after", right="before")
        b = mk(f"p{i}", "real", "before", "left", left="before", right="after")
        rows += [a, b]
    rows += [mk("c0", "check", "good", "left"), mk("s0", "check_same", "tie", "tie")]
    kept, st = reduce_rows(rows)
    cases.append(("side-picker discarded", len(kept) == 0, 0, report(st)))

    # 3. Tie handling: both tie kept, half-tie discarded.
    rows = pair("p0", "tie", "tie") + pair("p1", "tie", "after")
    rows += [mk("c0", "check", "good", "left"), mk("s0", "check_same", "tie", "tie")]
    kept, st = reduce_rows(rows)
    ok = len(kept) == 1 and kept[0]["chosen"] == "tie"
    cases.append(("tie rules", ok, 0, report(st)))

    # 4. Same-image checks mostly failed -> session voided (exit 1).
    rows = sum([pair(f"p{i}", "after", "after") for i in range(5)], [])
    rows += [mk(f"s{i}", "check_same", "after", "left") for i in range(3)]
    rows += [mk("s9", "check_same", "tie", "tie")]
    kept, st = reduce_rows(rows)
    cases.append(("same-image gate fires", True, 1, report(st)))

    # 5. Malformed: a pair with only one presentation is not silently kept.
    rows = pair("p0", "after", "after") + [mk("p1", "real", "after", "left")]
    rows += [mk("c0", "check", "good", "left")]
    kept, st = reduce_rows(rows)
    cases.append(("lone presentation dropped",
                  len(kept) == 1 and st["malformed"] == ["p1"], 0, report(st)))

    print("\n" + "=" * 64)
    for name, ok, want, got in cases:
        verdict = "OK" if (ok and got == want) else "FAIL"
        if verdict == "FAIL":
            bad += 1
        print(f"[{verdict}] {name}  (exit {got}, expected {want})")
    print(f"selftest {len(cases)} cases, {bad} failed")
    return 1 if bad else 0


def simulate(html: Path, out_dir: Path):
    """Drive the real page end to end, imitating task_template.html's save().

    The selftest above exercises the reduction rules on synthetic rows. It
    does NOT prove that the page actually exports rows this script can read
    -- that gap is exactly what round 65 found in the seam study, where the
    criteria were tested but the export format never was. So: parse the real
    ITEMS array, answer it under two annotator models, write the CSV with the
    same columns and the same order as save(), and read it back.
    """
    import json
    import random
    import re as _re

    items = json.loads(_re.search(r"const ITEMS = (\[.*?\]);",
                                  html.read_text(encoding="utf-8"), _re.S).group(1))
    # These must match the header emitted by task_template.html's save().
    head = FIELDS

    def export(name, answer):
        rng = random.Random(0)
        rows = []
        for i, it in enumerate(items):
            c = answer(it, rng)
            rows.append({"idx": i, "material": it["material"], "kind": it["kind"],
                         "struct": it.get("struct", ""), "stratum": it.get("stratum", ""),
                         "left": it["left"], "right": it["right"], "choice": c,
                         "chosen": "tie" if c == "tie" else (it["left"] if c == "left"
                                                             else it["right"]),
                         "ms": rng.randint(1500, 9000), "pair": it.get("pair", "")})
        p = out_dir / name
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=head)
            w.writeheader()
            w.writerows(rows)
        return p

    def honest(it, rng):
        """Prefers 'after'/'artist', and always gets the checks right."""
        if it["kind"] == "check_same":
            return "tie"
        if it["kind"] == "check":
            return "left" if it["left"] == "good" else "right"
        good = it.get("pair", "")
        target = "after" if "after" in (it["left"], it["right"]) else "artist"
        want = target if hash(good) % 10 < 8 else (
            it["left"] if it["right"] == target else it["right"])
        return "left" if it["left"] == want else "right"

    def side_picker(it, rng):
        """Round 67's failure mode: always press left."""
        return "left"

    print(f"\n--- end-to-end simulation on {html.name} ---")
    for name, model, expect in [("honest", honest, "most pairs kept"),
                                ("side_picker", side_picker, "all pairs discarded")]:
        p = export(f"sim_{name}.csv", model)
        with open(p, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        assert list(rows[0].keys()) == head, "export columns drifted from FIELDS"
        kept, st = reduce_rows(rows)
        print(f"\n[{name}] expect: {expect}")
        code = report(st)
        print(f"  kept {len(kept)} pairs, exit {code}")
        if name == "honest" and len(kept) == 0:
            print("  FAIL: an honest annotator should survive reduction")
            return 1
        if name == "side_picker" and len(kept) != 0:
            print("  FAIL: a pure side-picker must not survive reduction")
            return 1
    print("\nend-to-end simulation OK")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--simulate", type=Path,
                    help="drive a built v2 page end to end (export -> reduce)")
    a = ap.parse_args()

    if a.simulate:
        import tempfile
        raise SystemExit(simulate(a.simulate, Path(tempfile.mkdtemp())))
    if a.selftest:
        raise SystemExit(selftest())
    if not a.src:
        raise SystemExit("usage: reduce_paired_orders.py <study_x_v2.csv> -o reduced.csv")

    with open(a.src, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows or "pair" not in rows[0]:
        raise SystemExit(f"{a.src} has no 'pair' column; it is not a both-orders "
                         f"export. Use analyze_pair_study.py on it directly.")

    kept, stats = reduce_rows(rows)
    print(f"{a.src.name}: {len(rows)} rows")
    code = report(stats)

    out = a.out or a.src.with_name(a.src.stem + "_reduced.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for i, r in enumerate(kept + stats["blur"]):
            r = dict(r)
            r["idx"] = i
            w.writerow(r)
    print(f"wrote {out}  ({len(kept)} pairs + {len(stats['blur'])} checks)")
    print(f"next: python analysis/annotate/analyze_pair_study.py "
          f"<crop|ab60> {out}")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
