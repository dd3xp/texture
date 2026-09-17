"""(M47) 盲判读器：把 (M46) 的 `P_sel`（挑调色板，6.537 KID）再拆成一条四级阶梯。

    palette=xmodal → best5 → best30 → best_bank → palette=real

每一级差一个**可操作的杠杆**：
  S1 = xmodal  − best5      在正式配置那 5 个候选里挑对了没有   → 造重排器
  S2 = best5   − best30     重排候选从 5 扩到 30 值不值           → 放宽 topk
  S3 = best30  − best_bank  名字预筛 n_name=30 丢了多少           → 换/放宽检索
  S4 = best_bank − real     整个记忆库里根本没有这张调色板       → 得合成调色板

判据写于任何读数之前。thr = 4.656（(M33) m=1 KID 噪声下限），⛔ 小于 thr 一律只许读成"测不出"。
⚠ best* 三行全部用到目标瓦片自己的调色板 ＝ 不可达上界，只用于**定位**下一轮造哪个部件。

用法：python m47_read_split.py --json <diag json>    /    --selftest
"""
import argparse
import json
import sys

THR = 4.656                       # (M33) m=1 KID_x1e3 噪声下限
M46 = {"palette=xmodal": 13.431, "palette=real": 6.894}   # (M46) 已发表读数，用作复现操作检验
TOL = 0.5                         # 生成管线逐像素确定 ⇒ 同种子应逐位复现，留 0.5 余量
STEPS = [("S1", "palette=xmodal", "palette=best5", "GAP_PICK_IN_TOP5"),
         ("S2", "palette=best5", "palette=best30", "GAP_POOL_TOO_NARROW"),
         ("S3", "palette=best30", "palette=best_bank", "GAP_NAME_PREFILTER"),
         ("S4", "palette=best_bank", "palette=real", "GAP_BANK_LACKS")]
NEED = ["TRD", "palette=real", "palette=xmodal", "palette=best5", "palette=best30",
        "palette=best_bank", "real_half"]


def kid(d, row):
    return float(d[row]["KID_x1e3"])


def read(d):
    """返回 (判决, 详情)。判决 VOID_* 时 ⛔ 一条读数都不许引。"""
    miss = [r for r in NEED if r not in d]
    if miss:
        return "VOID_NO_DATA", {"missing": miss, "checked": len(NEED) - len(miss)}
    v = {r: kid(d, r) for r in NEED}
    det = {"kid": v, "thr": THR}
    det["dyn_range"] = v["TRD"] - v["real_half"]
    det["repro"] = {k: round(v[k] - e, 4) for k, e in M46.items()}
    det["P_sel"] = v["palette=xmodal"] - v["palette=real"]
    det["steps"] = {name: v[a] - v[b] for name, a, b, _ in STEPS}

    if det["dyn_range"] < THR:
        return "VOID_NO_DYNAMIC_RANGE", det
    bad = {k: x for k, x in det["repro"].items() if abs(x) > TOL}
    if bad:
        det["repro_failed"] = bad
        return "VOID_NOT_REPRODUCED", det
    # oracle 行按构造不该比它的超集差：某一级负得超过噪声下限 ⇒ 调色板距离与 KID 不同向，尺子不可用
    neg = {n: s for n, s in det["steps"].items() if s < -THR}
    if neg:
        det["negative_steps"] = neg
        return "VOID_METRIC_DISAGREE", det
    top = max(STEPS, key=lambda s: det["steps"][s[0]])
    det["largest"] = top[0]
    det["passed"] = [n for n, _, _, _ in STEPS if det["steps"][n] >= THR]
    if not det["passed"]:
        return "SPLIT_UNRESOLVED", det
    return top[3], det


CASES = [
    # (输入, 期望判决)
    ({}, "VOID_NO_DATA"),
    ({r: {"KID_x1e3": 1.0} for r in NEED if r != "TRD"}, "VOID_NO_DATA"),
    # 动态范围不够
    ({"TRD": 5.0, "real_half": 3.8, "palette=real": 6.894, "palette=xmodal": 13.431,
      "palette=best5": 12.0, "palette=best30": 10.0, "palette=best_bank": 7.0}, "VOID_NO_DYNAMIC_RANGE"),
    # 没复现 (M46)：xmodal 偏了
    ({"TRD": 38.9, "real_half": 3.8, "palette=real": 6.894, "palette=xmodal": 20.0,
      "palette=best5": 12.0, "palette=best30": 10.0, "palette=best_bank": 7.0}, "VOID_NOT_REPRODUCED"),
    # 没复现：palette=real 偏了
    ({"TRD": 38.9, "real_half": 3.8, "palette=real": 9.0, "palette=xmodal": 13.431,
      "palette=best5": 12.0, "palette=best30": 10.0, "palette=best_bank": 8.0}, "VOID_NOT_REPRODUCED"),
    # 容差边界：刚好 0.5 内 → 不 VOID
    ({"TRD": 38.9, "real_half": 3.8, "palette=real": 7.394, "palette=xmodal": 13.0,
      "palette=best5": 13.0, "palette=best30": 13.0, "palette=best_bank": 13.0}, "GAP_BANK_LACKS"),
    # 某一级显著为负 ⇒ 距离度量与 KID 不同向
    ({"TRD": 38.9, "real_half": 3.8, "palette=real": 6.894, "palette=xmodal": 13.431,
      "palette=best5": 20.0, "palette=best30": 10.0, "palette=best_bank": 7.0}, "VOID_METRIC_DISAGREE"),
    # 小的负值（噪声内）不触发 VOID
    ({"TRD": 38.9, "real_half": 3.8, "palette=real": 6.894, "palette=xmodal": 13.431,
      "palette=best5": 15.0, "palette=best30": 8.0, "palette=best_bank": 7.0}, "GAP_POOL_TOO_NARROW"),
    # 四级全在噪声里
    ({"TRD": 38.9, "real_half": 3.8, "palette=real": 6.894, "palette=xmodal": 13.431,
      "palette=best5": 11.4, "palette=best30": 9.4, "palette=best_bank": 7.9}, "SPLIT_UNRESOLVED"),
    # S1 最大且过门
    ({"TRD": 38.9, "real_half": 3.8, "palette=real": 6.894, "palette=xmodal": 13.431,
      "palette=best5": 8.0, "palette=best30": 7.5, "palette=best_bank": 7.2}, "GAP_PICK_IN_TOP5"),
    # S3 最大且过门
    ({"TRD": 38.9, "real_half": 3.8, "palette=real": 6.894, "palette=xmodal": 13.431,
      "palette=best5": 13.0, "palette=best30": 12.6, "palette=best_bank": 7.0}, "GAP_NAME_PREFILTER"),
    # S4 最大且过门（库里没货）
    ({"TRD": 38.9, "real_half": 3.8, "palette=real": 6.894, "palette=xmodal": 13.431,
      "palette=best5": 13.2, "palette=best30": 13.0, "palette=best_bank": 12.5}, "GAP_BANK_LACKS"),
    # 恰好等于 thr 算过（0.0002 纪律：只多不少）
    ({"TRD": 38.9, "real_half": 3.8, "palette=real": 6.894, "palette=xmodal": 13.431,
      "palette=best5": 13.431 - THR, "palette=best30": 8.0, "palette=best_bank": 7.4},
     "GAP_PICK_IN_TOP5"),
    # 差 0.001 就是没过（且其余更小）→ 未判定
    ({"TRD": 38.9, "real_half": 3.8, "palette=real": 6.894, "palette=xmodal": 13.431,
      "palette=best5": 13.431 - THR + 0.001, "palette=best30": 8.4, "palette=best_bank": 7.4},
     "SPLIT_UNRESOLVED"),
    # 动态范围恰好等于 thr → 不 VOID
    ({"TRD": 3.8 + THR, "real_half": 3.8, "palette=real": 6.894, "palette=xmodal": 13.431,
      "palette=best5": 7.4, "palette=best30": 7.2, "palette=best_bank": 7.0}, "GAP_PICK_IN_TOP5"),
    # VOID 优先级：动态范围不够时即使别的也坏，仍报动态范围
    ({"TRD": 4.0, "real_half": 3.8, "palette=real": 99.0, "palette=xmodal": 99.0,
      "palette=best5": 1.0, "palette=best30": 1.0, "palette=best_bank": 1.0}, "VOID_NO_DYNAMIC_RANGE"),
]


def selftest():
    ok = 0
    for i, (raw, want) in enumerate(CASES):
        d = {k: (v if isinstance(v, dict) else {"KID_x1e3": v}) for k, v in raw.items()}
        got, _ = read(d)
        if got == want:
            ok += 1
        else:
            print(f"  case {i}: 期望 {want}, 实得 {got}")
    print(f"selftest {ok}/{len(CASES)}")
    return ok == len(CASES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    with open(a.json, encoding="utf-8") as f:       # Windows 默认 GBK，必须显式 utf-8
        d = json.load(f)
    verdict, det = read(d)
    payload = {"verdict": verdict, **det}
    if a.out:                                        # 落盘一律在打印之前
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1, ensure_ascii=False)
    print("judgement:", verdict)
    for k, v in det.get("steps", {}).items():
        lab = next(s[3] for s in STEPS if s[0] == k)
        print(f"  {k} {lab:<22} {v:+.3f}  {'过' if v >= THR else '【噪声内】'}")
    if "P_sel" in det:
        print(f"  P_sel(xmodal-real) {det['P_sel']:+.3f}   dyn_range {det['dyn_range']:+.3f}"
              f"   repro {det['repro']}")
    if a.out:
        print("->", a.out)


if __name__ == "__main__":
    main()
