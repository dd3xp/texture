"""(M52) 判读器：oracle 调色板 vs 正式配置这条**上界臂**，判官看不看得见「挑调色板」这条轴。

判据在 docs/arch_progress.md 的 (M52) 预注册节里写死，本文件是它的机械实现，
**盲写**（写于判官臂启动之前，未见任何胜率）。跑之前先 `--selftest`。

    python analysis/arch/m52_read_axis.py --selftest
    python analysis/arch/m52_read_axis.py --judge <judge_full_*.json> --decomp <m52_decompose_s0.json> \
        --dump <dump 根目录> --out experiments/m52_verdict.json

落盘一律挪到打印之前；打印里不用非 GBK 字形。
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
from exact import binom_test, jeffreys          # noqa: E402

A_ROW, B_ROW = "palette=real", "palette=xmodal"
A_DIR, B_DIR = "pal_real", "pal_xmodal"
# (OP2) 必须复现的 (M50) seed0 读数（KID x1e3）
EXPECT_KID = {"palette=real": 6.894, "palette=xmodal": 13.431,
              "palette=rr_trdpal": 9.876, "TRD": 38.909}
KID_TOL = 0.01
MIN_SLUGS = 115
FLOOR = 21 / 118
RATE_LINE = 0.65
ALPHA = 0.05
MAX_API_FAIL = 0.05
MIN_DIFF = 0.5


def read_dump(dump: Path):
    """返回 (A 的 slug 集合, B 的 slug 集合, 同 slug 两图不同的比例)。dump 为 None 时返回 None。"""
    if dump is None:
        return None
    from PIL import Image
    import numpy as np
    da, db = dump / A_DIR / "16", dump / B_DIR / "16"
    if not da.is_dir() or not db.is_dir():
        return set(), set(), 0.0
    sa = {p.name for p in da.glob("*.png")}
    sb = {p.name for p in db.glob("*.png")}
    both = sorted(sa & sb)
    diff = 0
    for nm in both:
        x = np.asarray(Image.open(da / nm).convert("RGB"))
        y = np.asarray(Image.open(db / nm).convert("RGB"))
        diff += int(x.shape != y.shape or not (x == y).all())
    return sa, sb, (diff / len(both) if both else 0.0)


def decide(judge, decomp, dumpinfo):
    """judge: judge_full JSON dict; decomp: 拆解 JSON dict; dumpinfo: (sa, sb, diff_rate)。
    返回 (verdict, detail)。任一操作检验不过即 VOID_*，判决与操作检验的顺序在预注册里写死。"""
    d = {"ops": {}}
    # (OP1) 料齐：两边 slug 集合完全相同且够多
    if dumpinfo is None or judge is None or decomp is None:
        d["ops"]["OP1"] = False
        return "VOID_NO_DATA", d
    sa, sb, diff_rate = dumpinfo
    d["n_slug_a"], d["n_slug_b"] = len(sa), len(sb)
    d["ops"]["OP1"] = bool(sa == sb and len(sa) >= MIN_SLUGS)
    if not d["ops"]["OP1"]:
        return "VOID_NO_DATA", d
    # (OP2) 是不是那次拆解的图：四行 KID 复现 (M50) seed0
    got = {}
    for k, v in EXPECT_KID.items():
        row = decomp.get(k)
        got[k] = None if not isinstance(row, dict) else row.get("KID_x1e3")
    d["kid"] = got
    d["kid_expect"] = EXPECT_KID
    d["ops"]["OP2"] = all(got[k] is not None and abs(got[k] - v) < KID_TOL
                          for k, v in EXPECT_KID.items())
    if not d["ops"]["OP2"]:
        return "VOID_NOT_SAME_RUN", d
    # (OP3) 两边不是同一张图
    d["diff_rate"] = diff_rate
    d["ops"]["OP3"] = bool(diff_rate > MIN_DIFF)
    if not d["ops"]["OP3"]:
        return "VOID_SAME_IMAGE", d
    # (OP4) API
    n_pair = judge["decided"] + judge["inconsistent"] + judge["api_fail"]
    d["n_pair"] = n_pair
    d["api_fail_rate"] = judge["api_fail"] / n_pair if n_pair else 1.0
    d["ops"]["OP4"] = bool(n_pair > 0 and d["api_fail_rate"] <= MAX_API_FAIL)
    if not d["ops"]["OP4"]:
        return "VOID_API", d
    # (OP5) 可解率高于地板
    d["resolve_rate"] = judge.get("resolve_rate")
    d["p_vs_floor"] = judge.get("p_vs_floor")
    d["ops"]["OP5"] = bool(d["resolve_rate"] is not None and d["p_vs_floor"] is not None
                           and d["resolve_rate"] > FLOOR and d["p_vs_floor"] < ALPHA)
    if not d["ops"]["OP5"]:
        return "VOID_UNRESOLVABLE", d
    # 主判据
    w, n = judge["a_wins"], judge["decided"]
    d["a_wins"], d["decided"] = w, n
    d["rate"] = w / n if n else float("nan")
    d["p"] = binom_test(w, n) if n else float("nan")
    d["jeffreys"] = list(jeffreys(w, n)) if n else [float("nan")] * 2
    if d["rate"] < RATE_LINE:
        return "AXIS_TOO_SMALL", d
    if d["p"] < ALPHA:
        return "AXIS_VISIBLE", d
    return "VOID_UNDERPOWERED", d


def _j(w, n, inc=0, fail=0, rr=0.62, pf=1e-9):
    return {"a_wins": w, "decided": n, "inconsistent": inc, "api_fail": fail,
            "resolve_rate": rr, "p_vs_floor": pf}


def selftest():
    ok_dump = ({f"s{i}.png" for i in range(124)}, {f"s{i}.png" for i in range(124)}, 0.9)
    ok_dec = {k: {"KID_x1e3": v} for k, v in EXPECT_KID.items()}
    cases = [
        # 料不齐
        (None, None, None, "VOID_NO_DATA"),
        (_j(80, 120), ok_dec, (set(), set(), 0.0), "VOID_NO_DATA"),
        (_j(80, 120), ok_dec, ({"a.png"}, {"a.png"}, 0.9), "VOID_NO_DATA"),          # 太少
        (_j(80, 120), ok_dec, (ok_dump[0], {f"s{i}.png" for i in range(123)}, 0.9),
         "VOID_NO_DATA"),                                                            # 集合不同
        # OP2
        (_j(80, 120), {}, ok_dump, "VOID_NOT_SAME_RUN"),
        (_j(80, 120), {**ok_dec, "TRD": {"KID_x1e3": 38.93}}, ok_dump, "VOID_NOT_SAME_RUN"),
        (_j(80, 120), {**ok_dec, "palette=real": {"KID_x1e3": 6.894 + 0.02}}, ok_dump,
         "VOID_NOT_SAME_RUN"),
        (_j(80, 120), {**ok_dec, "palette=real": {"KID_x1e3": 6.894 + 0.005}}, ok_dump,
         "AXIS_VISIBLE"),                                                            # 容差内
        # OP3
        (_j(80, 120), ok_dec, (ok_dump[0], ok_dump[1], 0.5), "VOID_SAME_IMAGE"),
        (_j(80, 120), ok_dec, (ok_dump[0], ok_dump[1], 0.0), "VOID_SAME_IMAGE"),
        (_j(80, 120), ok_dec, (ok_dump[0], ok_dump[1], 0.51), "AXIS_VISIBLE"),
        # OP4
        (_j(80, 120, inc=0, fail=7), ok_dec, ok_dump, "VOID_API"),                   # 7/127 > 5%
        (_j(80, 120, inc=0, fail=6), ok_dec, ok_dump, "AXIS_VISIBLE"),               # 6/126 = 4.8%
        (_j(0, 0, 0, 0), ok_dec, ok_dump, "VOID_API"),
        # OP5
        (_j(80, 120, rr=FLOOR, pf=1e-9), ok_dec, ok_dump, "VOID_UNRESOLVABLE"),
        (_j(80, 120, rr=0.62, pf=0.05), ok_dec, ok_dump, "VOID_UNRESOLVABLE"),
        (_j(80, 120, rr=0.62, pf=0.049), ok_dec, ok_dump, "AXIS_VISIBLE"),
        # 主判据
        (_j(78, 120), ok_dec, ok_dump, "AXIS_VISIBLE"),                              # 65.0%
        (_j(77, 120), ok_dec, ok_dump, "AXIS_TOO_SMALL"),                            # 64.2%
        (_j(60, 120), ok_dec, ok_dump, "AXIS_TOO_SMALL"),                            # 50%
        (_j(74, 120), ok_dec, ok_dump, "AXIS_TOO_SMALL"),                            # 61.7% 显著但不够
        (_j(40, 120), ok_dec, ok_dump, "AXIS_TOO_SMALL"),                            # 反向也走这支
        (_j(13, 20), ok_dec, ok_dump, "VOID_UNDERPOWERED"),                          # 65% 但 p=0.26
        (_j(7, 10), ok_dec, ok_dump, "VOID_UNDERPOWERED"),                           # 70% 但 p=0.34
        (_j(124, 124), ok_dec, ok_dump, "AXIS_VISIBLE"),
    ]
    n_ok = 0
    for i, (j, dec, dmp, want) in enumerate(cases):
        got, _ = decide(j, dec, dmp)
        assert got == want, f"case {i}: 期望 {want}，得到 {got}"
        n_ok += 1
    # 几条不依赖 decide 的常量检验
    assert abs(binom_test(78, 120) - binom_test(78, 120)) == 0
    n_ok += 1
    assert 78 / 120 >= RATE_LINE > 77 / 120
    n_ok += 1
    assert FLOOR < 0.2 and MIN_SLUGS <= 124
    n_ok += 1
    assert set(EXPECT_KID) == {"palette=real", "palette=xmodal", "palette=rr_trdpal", "TRD"}
    n_ok += 1
    assert A_ROW in EXPECT_KID and B_ROW in EXPECT_KID
    n_ok += 1
    print(f"[selftest] {n_ok}/{n_ok} 通过")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--judge", type=Path, default=None)
    ap.add_argument("--decomp", type=Path, default=None)
    ap.add_argument("--dump", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/m52_verdict.json")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    judge = (json.loads(a.judge.read_text(encoding="utf-8"))
             if a.judge and a.judge.exists() else None)
    decomp = (json.loads(a.decomp.read_text(encoding="utf-8"))
              if a.decomp and a.decomp.exists() else None)
    dumpinfo = read_dump(a.dump if a.dump and a.dump.is_dir() else None)
    checked = {"judge": bool(judge), "decomp": bool(decomp), "dump": dumpinfo is not None}
    verdict, detail = decide(judge, decomp, dumpinfo)
    out = {"verdict": verdict, "checked": checked, "detail": detail,
           "lines": {"rate": RATE_LINE, "alpha": ALPHA, "floor": FLOOR,
                     "kid_tol": KID_TOL, "min_slugs": MIN_SLUGS,
                     "max_api_fail": MAX_API_FAIL, "min_diff": MIN_DIFF}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已查：judge={checked['judge']} decomp={checked['decomp']} dump={checked['dump']}")
    if "rate" in detail:
        print(f"oracle 胜 {detail['a_wins']}/{detail['decided']} = {detail['rate']:.1%}  "
              f"p={detail['p']:.3g}  可解率={detail['resolve_rate']:.1%}")
    print(f"判决：{verdict}  ->  {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
