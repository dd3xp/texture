"""从已入库的 JSON 复算三个**只能在 GPU 上产出**的论文数字，并逐条对照正文。

为什么需要这个脚本：`second_generator.py`、`controlnet_units.py`、
`point_sample_iso.py` 都要 diffusers + CUDA 才跑得出数，它们把结果写进 JSON
后**只在运行时打印一次统计**。于是净克隆（以及审稿人）拿到代码也验不了
B23/B24/B25 三段正文——除非把 JSON 入库、并有一条不依赖 GPU 的复算路径。
这个脚本就是那条路径：只读 JSON，不导入 torch。

对照的正文位置（`paper/main.tex`）：
  B24 第二个生成器  §4 第 421-424 行 + 附录 857-869 行
  B25 ControlNet    附录 876-884 行
  B23 点采样        §5 第 918-924 行

⚠ **口径警告（本脚本存在的第二个理由）**：`controlnet_units.py` 的中位数
把「纯提示」和「条件」各自过滤 `units>0` 后**分别**取中位，两个中位落在
**不同的材质集合**上（条件组多出两个纯提示没检出周期的材质）。
正文 876 行照抄了那对数，所以那句「从 27.0 降到 8.1」是**跨集合**比较。
配对集合（两边都检出周期的 10 个）上实为 27.0 → 15.8。
本脚本两种口径都打印，不替用户做取舍。

无 scipy：二项检验走 `analysis/exact.py` 的对数空间精确实现。
"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))

from exact import binom_test  # noqa: E402

ok = True


def check(label, got, want, tol=0.0):
    global ok
    hit = abs(got - want) <= tol if isinstance(want, float) else got == want
    ok = ok and hit
    print(f"  {'OK  ' if hit else 'FAIL'} {label}: 复算 {got}  正文 {want}")


def quartiles(xs):
    """n=41 时四分位落在数据点上；用 numpy/statistics 的 inclusive 口径。"""
    return statistics.quantiles(sorted(xs), n=4, method="inclusive")


def second_generator(path):
    print("== B24 第二个生成器（SD 1.5 @512）==  正文 §4 + 附录")
    recs = json.loads(path.read_text(encoding="utf-8"))
    got = [r["units"] for r in recs if r["units"] > 0]
    q1, med, q3 = quartiles(got)
    check("周期检出", f"{len(got)}/{len(recs)}", "41/42")
    check("单元数中位", round(med, 1), 16.8, 0.05)
    check("IQR 下", round(q1, 1), 7.3, 0.05)
    check("IQR 上", round(q3, 1), 26.3, 0.05)
    check("相对真人 3.2 的倍数", round(med / 3.2, 1), 5.2, 0.05)
    check("相对 SDXL@512 13.8 的倍数", round(med / 13.8, 1), 1.2, 0.05)
    check("低于 2× 真人惯例(6.4)的提示词", sum(1 for v in got if v < 6.4), 9)


def controlnet(p08, p10):
    print("\n== B25 ControlNet ==  正文附录")
    for tag, path in (("scale 0.8", p08), ("scale 1.0", p10)):
        recs = json.loads(path.read_text(encoding="utf-8"))
        pl = [r["units_plain"] for r in recs if r["units_plain"] > 0]
        co = [r["units_cond"] for r in recs if r["units_cond"] > 0]
        both = [(r["units_plain"], r["units_cond"]) for r in recs
                if r["units_plain"] > 0 and r["units_cond"] > 0]
        down = sum(1 for a, b in both if b < a)
        print(f"  {tag}：")
        print(f"    脚本口径（跨集合）  纯提示中位 {statistics.median(pl):.1f}"
              f"（n={len(pl)}） -> 条件中位 {statistics.median(co):.1f}（n={len(co)}）")
        print(f"    配对口径（同集合）  纯提示中位 "
              f"{statistics.median([a for a, _ in both]):.1f} -> 条件中位 "
              f"{statistics.median([b for _, b in both]):.1f}（n={len(both)}）")
        print(f"    操作检验：变小 {down}/{len(both)}，"
              f"符号检验 p={binom_test(down, len(both)):.3g}")
    # 只有主判据（操作检验）进正文判读，两个尺度都不过，路线据此关闭。
    r08 = json.loads(p08.read_text(encoding="utf-8"))
    r10 = json.loads(p10.read_text(encoding="utf-8"))
    for tag, recs, w, n in (("0.8", r08, 6, 10), ("1.0", r10, 5, 9)):
        both = [(r["units_plain"], r["units_cond"]) for r in recs
                if r["units_plain"] > 0 and r["units_cond"] > 0]
        d = sum(1 for a, b in both if b < a)
        check(f"scale {tag} 变小材质数", f"{d}/{len(both)}", f"{w}/{n}")
    ex = {r["material"]: (r["units_plain"], r["units_cond"]) for r in r10}
    check("brick wall @1.0 条件后", round(ex["brick wall"][1], 1), 2.8, 0.05)
    check("stone brick wall @1.0 条件后", round(ex["stone brick wall"][1]), 102)


def point_sample(path):
    print("\n== B23 点采样 ==  正文 §5")
    recs = json.loads(path.read_text(encoding="utf-8"))
    box = statistics.median(r["spread_box"] for r in recs)
    pnt = statistics.median(r["spread_point"] for r in recs)
    up = sum(1 for r in recs if r["spread_point"] > r["spread_box"])
    check("跨度中位 box", round(box, 3), 0.109, 5e-4)
    check("跨度中位 point", round(pnt, 3), 0.480, 5e-4)
    check("逐材质上升", f"{up}/{len(recs)}", "29/29")
    check("符号检验 p", float(f"{binom_test(up, len(recs)):.1e}"), 3.7e-9, 5e-11)
    dec = [r for r in recs if r["vlm"] in ("box", "point")]
    win = sum(1 for r in dec if r["vlm"] == "point")
    print(f"  判官偏好点采样 {win}/{len(dec)} = {win / len(dec):.0%}，"
          f"p={binom_test(win, len(dec)):.2g}（弃用不一致 "
          f"{len(recs) - len(dec)} 对）")


def main():
    exp = ROOT / "experiments"
    second_generator(exp / "second_generator.json")
    controlnet(exp / "controlnet_units_s08.json", exp / "controlnet_units_s10.json")
    point_sample(exp / "point_sample_iso.json")
    print("\n全部对上正文" if ok else "\n有对不上的数字，见上面的 FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
