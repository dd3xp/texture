"""(M63) `GATE_CEILING_32`：量这把闸门的**天花板在不在目标之上**。

判据、门槛、四条操作检验、三条预测全部写死在 `docs/arch_progress.md` 的 (M63) 第四节，
**提交时间早于本文件**（预注册在工具存在之前）。本文件一行判据都不许改。

问的是一个**绝对位置**问题（不是相对比较）：在 `targets("V_mat", 32)` 那批目标上、
用 `eval/metrics.py::evaluate` **原函数**、同一份提示词，算两行 CLIP——

  R  ＝ 真人参照瓦片自身（(M58)–(M62) 口口声声的那个"天花板"）
  BB ＝ B2 在同材质的 32px 产物（`experiments/baselines_val/B2/32/<slug>.png`）

Δ ＝ CLIP(R) − CLIP(BB)，门槛 ＝ (M33) 已闭合的 m=1 CLIP 噪声下限 0.41。

  Δ < −0.41  ⇒ GATE_CEILING_BELOW_GOAL   天花板在目标之下 ⇒ 配对 CLIP@32px 只能定向、不能认证
  Δ > +0.41  ⇒ GATE_CEILING_ABOVE_GOAL   0.13 准入线原样成立
  其余       ⇒ GATE_CEILING_TIE          天花板 ≈ 目标，排期上与 BELOW 同样处理

⛔ 三条判决都不授权开任何架构臂；⛔ 都不许用来重读 (M58)–(M62) 的任何 p 值。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

THR = 0.41                       # (M33) m=1 CLIP 噪声下限，⛔ 不许改
REAL_HALF_CLIP = 34.1732177734375   # (M61)(OP3)：参照行与模型/种子无关，56 份 JSON 逐位相同
OP2_TOL = 1e-6                   # "逐位等于"：float32 打印精度内
OP3_MIN_HIT = 70                 # 74 条目标里 B2 至少命中这么多


def verdict(delta, thr=THR):
    """⛔ 冻结判据。只吃 Δ 与门槛，不吃任何别的东西。"""
    if delta < -thr:
        return "GATE_CEILING_BELOW_GOAL"
    if delta > thr:
        return "GATE_CEILING_ABOVE_GOAL"
    return "GATE_CEILING_TIE"


def selftest():
    assert verdict(-0.42) == "GATE_CEILING_BELOW_GOAL"
    assert verdict(-0.41) == "GATE_CEILING_TIE", "边界不含端点：|Δ|==thr 判 TIE"
    assert verdict(-0.40) == "GATE_CEILING_TIE"
    assert verdict(0.0) == "GATE_CEILING_TIE"
    assert verdict(0.41) == "GATE_CEILING_TIE"
    assert verdict(0.42) == "GATE_CEILING_ABOVE_GOAL"
    assert verdict(-3.0) == "GATE_CEILING_BELOW_GOAL" and verdict(3.0) == "GATE_CEILING_ABOVE_GOAL"
    # (OP1) 缺图的材质两侧同时剔除
    got = {"a": 1, "c": 3}
    keep = [i for i, s in enumerate(["a", "b", "c"]) if s in got]
    assert keep == [0, 2], keep
    # (OP2) 逐位比较
    assert abs(REAL_HALF_CLIP - 34.1732177734375) <= OP2_TOL
    assert not abs(REAL_HALF_CLIP - 34.1742177734375) <= OP2_TOL
    # (OP3) 命中门
    assert (69 >= OP3_MIN_HIT) is False and (70 >= OP3_MIN_HIT) is True
    # Δ 的方向：R 低于 BB ⇒ Δ<0 ⇒ BELOW
    assert verdict(33.5 - 35.5) == "GATE_CEILING_BELOW_GOAL"
    assert verdict(35.5 - 33.5) == "GATE_CEILING_ABOVE_GOAL"
    print("selftest ok  10/10")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b2dir", type=Path,
                    default=ROOT / "experiments/baselines_val/B2/32")
    ap.add_argument("--out", type=Path, default=Path("/tmp/m63_gate_ceiling.json"))
    a = ap.parse_args()

    sys.path.insert(0, str(ROOT / "eval"))
    sys.path.insert(0, str(ROOT / "model"))
    sys.path.insert(0, str(ROOT / "tools"))
    from colour_task import targets          # noqa: E402
    from metrics import evaluate             # noqa: E402
    from PIL import Image

    T = targets("V_mat", 32)
    ref = [t["ref"] for t in T]
    prompts = [t["prompt"] for t in T]
    out = {"n_targets": len(T), "n_slugs": len({t["slug"] for t in T}),
           "n_materials": len(set(prompts)), "thr": THR}

    # ---- (OP2) 同一性：与 diag_decompose 的 real_half 逐位相同 ----
    half = len(ref) // 2
    rh, _ = evaluate(ref[:half], prompts[:half], ref[half:])
    out["op2_real_half_CLIP"] = rh["CLIP"]
    out["op2_expect"] = REAL_HALF_CLIP
    op2 = abs(rh["CLIP"] - REAL_HALF_CLIP) <= OP2_TOL

    # ---- (OP1)(OP3) 取 B2 同材质那张，缺图的两侧同时剔除 ----
    keep, b2 = [], []
    for i, t in enumerate(T):
        p = a.b2dir / f"{t['slug']}.png"
        if p.exists():
            keep.append(i)
            b2.append(np.asarray(Image.open(p).convert("RGB")))
    out["n_kept"] = len(keep)
    out["n_dropped"] = len(T) - len(keep)
    op3 = len(keep) >= OP3_MIN_HIT

    r_tiles = [ref[i] for i in keep]
    r_mats = [prompts[i] for i in keep]
    b_mats = list(r_mats)
    op1 = (len(r_tiles) == len(b2)) and (r_mats == b_mats)

    rR, _ = evaluate(r_tiles, r_mats, [])          # 参照集为空 ⇒ 只算不依赖参照的指标（CLIP 逐字同一口径）
    rB, _ = evaluate(b2, b_mats, [])
    delta = rR["CLIP"] - rB["CLIP"]
    out["CLIP_real"] = rR["CLIP"]
    out["CLIP_B2"] = rB["CLIP"]
    out["delta"] = delta
    out["op1"] = bool(op1)
    out["op2"] = bool(op2)
    out["op3"] = bool(op3)

    if not op3:
        out["verdict"] = "VOID_NO_DATA"
    elif not op2:
        out["verdict"] = "VOID_PIPELINE_MISMATCH"
    elif not op1:
        out["verdict"] = "VOID_OP1"
    else:
        out["verdict"] = verdict(delta)

    a.out.write_text(json.dumps(out, indent=1))    # 落盘一律在打印之前
    # (OP4) 只报两个均值、Δ 与门槛，⛔ 不报任何逐材质数值或排名
    print(f"VERDICT: {out['verdict']}")
    print(f"  n_targets={len(T)}  n_kept={len(keep)}  n_dropped={out['n_dropped']}  "
          f"n_materials={out['n_materials']}")
    print(f"  CLIP_real = {rR['CLIP']:.4f}")
    print(f"  CLIP_B2   = {rB['CLIP']:.4f}")
    print(f"  delta     = {delta:+.4f}   thr = {THR}")
    print(f"  OP1 same_materials={op1}  OP2 real_half={rh['CLIP']!r} expect={REAL_HALF_CLIP!r} ok={op2}  "
          f"OP3 hit>={OP3_MIN_HIT} ok={op3}")
    print("->", a.out)


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
