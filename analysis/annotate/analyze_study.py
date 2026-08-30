"""D6：分析盲比结果。

主问题不是"总胜率多少"，而是**胜率随结构分怎么变**。
D5 观察到模型在均质材质上可用、在几何结构材质上输给基线；
若这个观察成立，胜率应当随结构分单调下降。
把它做成连续变量的检验，而不是先分类再比——
分类器（自相关峰强度）已证明抓不到砖块的二维错缝。

用法：
    python analyze_study.py --csv <标注导出的 csv>
    python analyze_study.py --simulate     # 用模拟数据自检脚本本身
"""

import argparse
import csv
import io
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def analyze_strata(real: list[dict]) -> None:
    """按 stratum 分层统计 模型 vs 基线。

    先验只对 42% 的材质加种子，效应只可能出现在 seeded 层；
    plain 层是对照，应当与之前无异。合并统计会把信号稀释掉。
    """
    mb = [r for r in real
          if sorted([r["left"], r["right"]]) == ["baseline", "model"]
          and r["choice"] != "tie" and r.get("stratum")]
    if not mb:
        return
    print(f"\n=== 分层：模型 vs 基线（n={len(mb)}）===")
    print(f"{'层':<26}{'n':>5}{'模型胜':>8}{'胜率':>9}{'双尾 p':>10}")
    print("-" * 60)
    for st, label in (("seeded", "有种子（先验作用层）"), ("plain", "无种子（对照层）")):
        sub = [r for r in mb if r["stratum"] == st]
        if not sub:
            continue
        w = sum(1 for r in sub if r["chosen"] == "model")
        n = len(sub)
        p = stats.binomtest(w, n, 0.5).pvalue
        print(f"{label:<26}{n:>5}{w:>8}{w/n:>8.0%}{p:>10.3f}")
    a = [r for r in mb if r["stratum"] == "seeded"]
    b = [r for r in mb if r["stratum"] == "plain"]
    if a and b:
        tab = [[sum(1 for r in a if r["chosen"] == "model"),
                sum(1 for r in a if r["chosen"] != "model")],
               [sum(1 for r in b if r["chosen"] == "model"),
                sum(1 for r in b if r["chosen"] != "model")]]
        pf = stats.fisher_exact(tab).pvalue
        print(f"\n两层胜率差异 Fisher p={pf:.3g}"
              f"  （先验若有效，seeded 应显著高于 plain）")
    print("\n判读（A4 探路的判据，n=47 时）：")
    print("  >70% → 方向对，值得扩到 D6 规模")
    print("  55–65% → 可能有效，需要更大样本确认")
    print("  ~50% → 先验未转化为人的偏好，不值得继续扩")


def simulate(n_mat: int = 24, seed: int = 0) -> list[dict]:
    """模拟一份标注，用来自检脚本。

    植入的真值：模型胜率随结构分从 0.75 线性降到 0.25。
    脚本若正确，应当把这个下降趋势检出来。
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_mat):
        st = 0.20 + 0.56 * i / max(n_mat - 1, 1)
        p_model = 0.75 - 0.50 * (st - 0.20) / 0.56
        for _ in range(1):
            win = rng.random() < p_model
            rows.append({"idx": len(rows), "material": f"m{i}", "kind": "real",
                         "struct": f"{st:.4f}",
                         "stratum": "seeded" if i % 2 == 0 else "plain",
                         "left": "model", "right": "baseline",
                         "choice": "left" if win else "right",
                         "chosen": "model" if win else "baseline"})
        if i % 3 == 0:
            rows.append({"idx": len(rows), "material": f"m{i}", "kind": "check",
                         "struct": f"{st:.4f}", "left": "good", "right": "blur",
                         "choice": "left", "chosen": "good"})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--bins", type=int, default=3)
    args = ap.parse_args()

    rows = simulate() if args.simulate else load(args.csv)
    if args.simulate:
        print("=== 自检模式：模拟数据，植入趋势为「胜率随结构分 0.75 → 0.25」===\n")

    chk = [r for r in rows if r["kind"] == "check"]
    real = [r for r in rows if r["kind"] == "real"]
    ok = sum(1 for r in chk if r["chosen"] == "good")
    print(f"标注 {len(rows)} 条：有效 {len(real)}，注意力检查 {ok}/{len(chk)} 正确")
    if chk and ok / len(chk) < 0.8:
        print("  ⚠ 注意力检查通过率偏低，结果可信度存疑")

    # --- 两两对决总胜率 ---
    print(f"\n{'对决':<26}{'胜':>5}{'负':>5}{'平':>5}{'胜率':>9}{'双尾 p':>10}")
    print("-" * 62)
    duel = defaultdict(lambda: [0, 0, 0])
    for r in real:
        a, b = sorted([r["left"], r["right"]])
        if r["choice"] == "tie":
            duel[(a, b)][2] += 1
        elif r["chosen"] == a:
            duel[(a, b)][0] += 1
        else:
            duel[(a, b)][1] += 1
    for (a, b), (w, l, t) in sorted(duel.items()):
        n = w + l
        p = stats.binomtest(w, n, 0.5).pvalue if n else float("nan")
        print(f"{a+' vs '+b:<26}{w:>5}{l:>5}{t:>5}"
              f"{(w/n if n else float('nan')):>8.0%}{p:>10.3f}   ({a} 的胜率)")

    if any(r.get("stratum") for r in real):
        analyze_strata(real)

    # --- 主分析：模型 vs 基线 的胜率随结构分怎么变 ---
    mb = [r for r in real
          if sorted([r["left"], r["right"]]) == ["baseline", "model"]
          and r["choice"] != "tie"]
    if not mb:
        print("\n没有 模型 vs 基线 的有效对，跳过分层分析")
        return
    st = np.array([float(r["struct"]) for r in mb])
    win = np.array([1 if r["chosen"] == "model" else 0 for r in mb])

    print(f"\n=== 主分析：模型 vs 基线，胜率随结构分的变化（n={len(mb)}）===")
    qs = np.quantile(st, np.linspace(0, 1, args.bins + 1))
    print(f"\n{'结构分区间':<22}{'n':>5}{'模型胜':>8}{'胜率':>9}")
    print("-" * 46)
    for i in range(args.bins):
        lo, hi = qs[i], qs[i + 1]
        sel = (st >= lo) & (st <= hi if i == args.bins - 1 else st < hi)
        if sel.sum() == 0:
            continue
        print(f"{f'{lo:.3f} – {hi:.3f}':<22}{int(sel.sum()):>5}"
              f"{int(win[sel].sum()):>8}{win[sel].mean():>8.0%}")

    rho, p_rho = stats.spearmanr(st, win)
    print(f"\n结构分 vs 模型是否获胜：Spearman ρ={rho:+.3f}  p={p_rho:.3g}")
    lo_sel, hi_sel = st <= np.median(st), st > np.median(st)
    if lo_sel.sum() and hi_sel.sum():
        tab = [[int(win[lo_sel].sum()), int((~win.astype(bool))[lo_sel].sum())],
               [int(win[hi_sel].sum()), int((~win.astype(bool))[hi_sel].sum())]]
        p_f = stats.fisher_exact(tab).pvalue
        print(f"低结构分半 胜率 {win[lo_sel].mean():.0%}  "
              f"高结构分半 胜率 {win[hi_sel].mean():.0%}  Fisher p={p_f:.3g}")

    print("\n判读：")
    if p_rho < 0.05 and rho < 0:
        print("  胜率随结构分显著下降 → D5 的观察成立："
              "路线在均质材质上可用，在几何结构上还差一步")
    elif p_rho < 0.05:
        print("  胜率随结构分显著上升 → 与 D5 观察相反，需要重新解释")
    else:
        print("  未检出显著趋势 → 样本量可能不足，不能据此下结论")


if __name__ == "__main__":
    main()
