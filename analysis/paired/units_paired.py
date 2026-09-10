"""剂量-反应的单元数中位（27.7 / 13.8 / 9.4）：跨集合口径 vs 配对口径。

**为什么写这个**（2026-09-10）：coherence 那轮末尾留下的教训——
「凡两组各自取中位再相除/相减得出的『差距』，必须先落到逐样本统计量上重测」——
只写在散文里，没人查过它还命中哪些**已发表**的数。已知两处：B25 的 27.0→8.1
（跨集合，仍在 main.tex 附录待用户拍板）与 coherence 的 0.67×/0.34×（已作废）。

`analysis/paired/fig_units.py:36` 是同一个形状：`period>0` 这个过滤是
**逐条件各自做的**，于是 1024/512/384 三个中位可能落在**三个不同的提示词子集**上。
它喂的是图 7 与 §5.4 的剂量-反应，比 B25 那句附录描述重要得多。

**判据（跑前写死，commit 后再运行）**

1. **自检先行**：本脚本必须先用 `fig_units.py` 的原口径**精确复现**已发表的
   27.7 / 13.8 / 9.4（容差 0.05）。复现不了 → 打印 REFUSE 并退出码 1，不予判读。
2. 三个条件的**提示词子集若完全相同** → 不存在跨集合问题，记「已核，无缺陷」，收工。
3. 若不同 → 在**三条件交集**上重算配对中位，并对每一对条件做**配对符号检验**
   （逐提示词比大小，平局弃用，精确二项）。
4. **判读只看方向与配对检验**，不看中位数掉了多少：
   - 配对检验方向与已发表的单调下降**一致且显著** → 结论稳固，中位数按配对口径更正即可；
   - 方向不一致或不显著 → 这就不只是口径问题，剂量-反应的单元数那条腿要重写。
5. 无论结论如何，**本脚本不改 `paper/`**（论文处于用户拍板冻结中）。

只读 JSON，不导 torch，净克隆可跑：`python analysis/paired/units_paired.py`
"""

import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# (canvas, json, 已发表的中位)
CONDS = [
    (1024, "experiments/crop_ctrl.json", 27.7),
    (512, "experiments/crop_render512.json", 13.8),
    (384, "experiments/crop_render384.json", 9.4),
]


def units(path, canvas):
    """与 fig_units.py:32-38 逐字同口径：period>0、按 prompt 去重、不限门触发。"""
    recs = json.loads((ROOT / path).read_text())
    seen = {}
    for r in recs:
        if r["period"] and r["period"] > 0 and r["prompt"] not in seen:
            seen[r["prompt"]] = canvas / r["period"]
    return seen


def sign_test(hi, lo):
    """hi/lo 为等长配对序列；返回 (下降数, 有效对数, 双侧精确 p)。"""
    down = sum(1 for a, b in zip(hi, lo) if b < a)
    up = sum(1 for a, b in zip(hi, lo) if b > a)
    n = down + up
    if n == 0:
        return down, n, 1.0
    k = min(down, up)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return down, n, min(1.0, 2 * tail)


def main():
    per_cond = {c: units(p, c) for c, p, _ in CONDS}

    print("== 自检：复现 fig_units.py 的已发表中位 ==")
    ok = True
    for canvas, _, published in CONDS:
        d = per_cond[canvas]
        med = statistics.median(d.values())
        hit = abs(med - published) <= 0.05
        ok &= hit
        print(f"  {canvas:>4}px  n={len(d):>2}  median={med:7.4f}  "
              f"已发表={published}  {'OK' if hit else 'MISMATCH'}")
    if not ok:
        print("REFUSE：自检未过，按判据不予判读。")
        return 1

    sets = {c: set(d) for c, d in per_cond.items()}
    inter = set.intersection(*sets.values())
    union = set.union(*sets.values())
    print("\n== 子集是否相同 ==")
    for canvas, _, _ in CONDS:
        extra = sorted(sets[canvas] - inter)
        print(f"  {canvas:>4}px 检出 {len(sets[canvas]):>2} / 并集 {len(union)}"
              f"{'；独有 ' + ', '.join(extra) if extra else ''}")
    if len(inter) == len(union):
        print("\n三条件子集完全相同 → 不存在跨集合问题。已核，无缺陷。")
        return 0

    print(f"\n三条件子集**不同**（交集 {len(inter)}）→ 按判据在交集上重算。")
    keys = sorted(inter)
    print("\n== 配对口径（同 %d 个提示词） ==" % len(keys))
    for canvas, _, published in CONDS:
        vals = [per_cond[canvas][k] for k in keys]
        print(f"  {canvas:>4}px  配对中位={statistics.median(vals):7.4f}   "
              f"（已发表跨集合 {published}）")

    print("\n== 配对符号检验（逐提示词） ==")
    for i in range(len(CONDS)):
        for j in range(i + 1, len(CONDS)):
            hi_c, lo_c = CONDS[i][0], CONDS[j][0]
            hi = [per_cond[hi_c][k] for k in keys]
            lo = [per_cond[lo_c][k] for k in keys]
            down, n, p = sign_test(hi, lo)
            print(f"  {hi_c}px → {lo_c}px：下降 {down}/{n}，p={p:.3g}")

    print("\n判读按判据第 4 条：方向一致且显著 → 口径更正；否则该腿要重写。")
    print("（本脚本不改 paper/。）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
