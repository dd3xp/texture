"""(M46) 盲写判读器：TRD 与真人的**剩余**分布差距还在哪一半（结构 / 挑调色板）。

写于任何读数存在之前（见 docs/arch_progress.md 本轮"预注册"一节）。跑完 `--selftest` 才允许上真数据。

## 为什么这一轮可以用分布指标，而配置审计那五轮不可以

⚠ (M37)–(M44) 反复量到：**KID/FID/FD/CLIP 在"配置差异"这个尺度上基本全瞎**
（CFG 0/8 过门、代次 0/8、(M44) 数据 5× 0/8），因为那些差普遍只有 1–3 KID，低于噪声下限。
⚑ 但**拆解尺度不是配置尺度**：2026-09-11 那次拆解读到的是 27.9 / 24.1 / 4.5 / 3.8 ——
跨度 24 KID，是下限 4.656 的 5 倍。同一把尺子在这个量级上是有分辨力的。
⛔ 因此本判读器的每一条判据都必须显式过 thr，且**小于 thr 一律只许读成"测不出"，不许读成"已到地板"**。

## 为什么挑这条路

⚑ 全项目**唯一**一个 ≥65% 的架构级效应（AB_pal 66%，p=8e-06，`W1_robust`）不是猜出来的：
它是先用 `eval/diag_decompose.py` 把差距定位到"配色"（84% 的超地板差距在那一半），
再照着这个定位造出来的部件。⇒ **"先定位差距、再造部件"是本项目已知唯一产出过承重部件的选杠杆法。**
⚠⚠ 但这条因果链的证据只有 **n=1**，⛔ 不许当成"定位到哪就一定能造出 ≥65% 的部件"。
⚠ 那次拆解跑在 **v2、还没有调色板记忆库**的模型上；装上之后差距搬到哪去了，全项目从未再问。本轮问这个。

## 读什么

四行**共用同一张 TRD 网格**，只换调色板（外加一行换网格），所以两轴是隔离的：

  TRD                模型自己的网格 + 模型自己的调色板
  struct=real        **真人**网格 + 模型自己的调色板   → 差距只来自配色
  palette=real       模型网格 + **真人**调色板         → 差距只来自结构（oracle 调色板）
  palette=retrieved  模型网格 + 老的同色数文本检索调色板
  palette=xmodal     模型网格 + **正式配置**那套跨模态检索调色板
  real_half          真人一半 vs 另一半               → 地板 F

主判据（全部用 KID_x1e3，thr = 4.656 = (M33) `FLOOR_AGREES` 的 m=1 下限）：

  S_gap = palette=real  - F            结构残差（把配色换成 oracle 之后还剩多少）
  P_sel = palette=xmodal - palette=real  **挑调色板**这一步相对 oracle 还差多少（结构held fixed）

⛔ P_sel 才是"还能不能靠调色板侧再捞一个部件"的那个量；⛔ 不许拿 struct=real 去回答它
（那一行换的是网格、且用的是模型自己的调色板头，正式配置根本不用那个头）。
"""
import argparse
import json
import sys
from pathlib import Path

THR = 4.656          # (M33) FLOOR_AGREES，m=1 的 KID_x1e3 噪声下限。⛔ 本轮一个字未改。
KEY = "KID_x1e3"
NEED = ["TRD", "struct=real", "palette=real", "palette=retrieved", "palette=xmodal", "real_half"]


def verdict(kid):
    """kid: 行名 -> KID_x1e3。返回 (判决, 明细 dict)。纯函数，无 IO。"""
    miss = [k for k in NEED if k not in kid]
    if miss:
        return "VOID_NO_DATA", {"missing": miss, "checked": len(NEED) - len(miss)}
    F = kid["real_half"]
    d = {
        "floor": F,
        "dyn_range": kid["TRD"] - F,                       # 操作检验：尺子在这批料上有没有动态范围
        "S_gap": kid["palette=real"] - F,                  # 结构残差
        "P_sel": kid["palette=xmodal"] - kid["palette=real"],   # 挑调色板相对 oracle 的残差
        "P_old": kid["palette=retrieved"] - kid["palette=real"],  # 老检索的同一个量（只作参考）
        "thr": THR,
    }
    # 操作检验：不过就 VOID，⛔ 不许往下读任何一条
    if d["dyn_range"] < THR:
        return "VOID_NO_DYNAMIC_RANGE", d
    s, p = d["S_gap"] >= THR, d["P_sel"] >= THR
    if s and p:
        d["larger"] = "structure" if d["S_gap"] > d["P_sel"] else "palette_selection"
        return "GAP_BOTH", d
    if s:
        return "GAP_STRUCTURE", d
    if p:
        return "GAP_PALETTE_SELECTION", d
    return "GAP_UNRESOLVED", d          # 两轴都测不出 ⇒ ⛔ 本轮不许用来选杠杆


ACTION = {
    "GAP_STRUCTURE": "下一个候选部件往**结构**侧造（配色侧已榨干，再修调色板买不到东西）",
    "GAP_PALETTE_SELECTION": "下一个候选部件往**挑调色板**侧造（oracle 调色板与检索到的之间还有可捞的量）",
    "GAP_BOTH": "两轴都还有量，按 larger 那一侧先造",
    "GAP_UNRESOLVED": "【禁】本轮不许用来选杠杆：两轴残差都在噪声里，这把尺子在当前配置上已分辨不出，另找工具",
    "VOID_NO_DYNAMIC_RANGE": "【禁】作废：尺子在这批料上连 TRD 与地板都分不开，任何一条读数都不许引用",
    "VOID_NO_DATA": "【禁】作废：料不全（见 missing / checked）",
}


def selftest():
    base = {"TRD": 30.0, "struct=real": 25.0, "palette=real": 5.0,
            "palette=retrieved": 12.0, "palette=xmodal": 11.0, "real_half": 4.0}
    cases = []

    def chk(name, kid, want):
        got = verdict(kid)[0]
        cases.append((name, got == want, f"{got} != {want}"))

    # 1) 结构残差 1.0 < thr、挑调色板残差 6.0 >= thr
    chk("palette_selection", base, "GAP_PALETTE_SELECTION")
    # 2) 结构残差大、调色板残差小
    k = dict(base, **{"palette=real": 15.0, "palette=xmodal": 16.0})
    chk("structure", k, "GAP_STRUCTURE")
    # 3) 两轴都过
    k = dict(base, **{"palette=real": 15.0, "palette=xmodal": 30.0})   # S_gap=11, P_sel=15
    chk("both", k, "GAP_BOTH")
    assert verdict(k)[1]["larger"] == "palette_selection", "larger 选错"
    k2 = dict(base, **{"palette=real": 25.0, "palette=xmodal": 35.0})  # S_gap=21, P_sel=10
    assert verdict(k2)[0] == "GAP_BOTH" and verdict(k2)[1]["larger"] == "structure", "larger 选错(2)"
    cases.append(("both_larger", True, ""))
    # 4) 两轴都测不出
    k = dict(base, **{"palette=xmodal": 5.5})
    chk("unresolved", k, "GAP_UNRESOLVED")
    # 5) 恰好贴线：>= thr 算过，差一点点算没过（(M14) 纪律：差 0.0002 也是没过）
    # ⚠ 用 0.0 作基准构造，避免 5.0+THR-5.0 的浮点误差自己把"恰好贴线"打成没过
    k = dict(base, **{"palette=real": 0.0, "palette=xmodal": THR})
    chk("exactly_thr_passes", k, "GAP_PALETTE_SELECTION")
    k = dict(base, **{"palette=real": 0.0, "palette=xmodal": THR - 1e-9})
    chk("just_below_thr_fails", k, "GAP_UNRESOLVED")
    # 6) 没有动态范围 → VOID，且**优先于**任何 GAP_*
    k = dict(base, **{"TRD": 4.0 + THR - 1e-9})
    chk("void_no_dynamic_range", k, "VOID_NO_DYNAMIC_RANGE")
    # 7) 缺行 → VOID_NO_DATA，且必须报"已查几项"（⛔ 不许拿空集冒充"量过没事"）
    for missing in NEED:
        k = {x: v for x, v in base.items() if x != missing}
        chk(f"void_missing_{missing}", k, "VOID_NO_DATA")
    assert verdict({})[1]["checked"] == 0
    cases.append(("void_empty_checked0", True, ""))
    # 8) 地板抬高时两个残差同步缩小（判据确实是相对地板/相对 oracle 的）
    k = dict(base, real_half=10.0, TRD=30.0)
    assert verdict(k)[1]["S_gap"] == -5.0
    cases.append(("floor_relative", True, ""))

    bad = [(n, m) for n, ok, m in cases if not ok]
    for n, ok, m in cases:
        print(("  ok  " if ok else "  FAIL") + f" {n}" + ("" if ok else f"  {m}"))
    print(f"selftest {len(cases) - len(bad)}/{len(cases)}")
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, help="diag_decompose 的产物")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.json:
        ap.error("要么 --selftest 要么 --json")
    raw = json.loads(a.json.read_text(encoding="utf-8"))     # Windows 默认 GBK，中文字段会崩
    kid = {k: v[KEY] for k, v in raw.items() if isinstance(v, dict) and KEY in v}
    v, d = verdict(kid)
    print("== (M46) 剩余差距拆解 ==")
    for k in NEED:
        print(f"  {k:<18} {KEY}={kid.get(k, float('nan')):.3f}")
    print(f"  thr(m=1)={THR}")
    for k in ("dyn_range", "S_gap", "P_sel", "P_old"):
        if k in d:
            print(f"  {k:<10} {d[k]:+.3f}   {'过' if abs(d[k]) >= THR else '在噪声里'}")
    print(f"判决: {v}")
    print(f"动作: {ACTION[v]}")
    print("【禁】任何 < thr 的量只许读成「测不出」，不许读成「已到地板」；")
    print("【禁】本轮只定位差距，不产生任何判官结论，也不改动任何已发表数字。")


if __name__ == "__main__":
    main()
