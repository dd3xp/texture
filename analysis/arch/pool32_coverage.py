"""32px 训练池对**测试材质**的覆盖盘点（零 GPU、零 API，只读已入库 JSON）。

**为什么问这个**：32px 是唯一还在的缺口（正式测试 TRD32_rr4 vs B2 = 83/202 = 41%，p=0.014），
而同一套东西 16px 胜（114/199 = 57%）、24px 平（94/188 = 50%）。账本已把"在 641 张的池子里
调旋钮"全部关闭，下一题是**换数据来源**——但在花力气弄新数据之前，得先知道现有池子到底缺什么：

- 缺的是**张数**（同样的材质，样本少）？
- 还是缺的是**材质覆盖**（测试要画的材质，32px 池里压根没有同名的真人样本）？

这两种"缺"要用完全不同的办法补，所以先量出来。⚠ 本脚本**一个胜负数字都不看**，
只数训练池与测试提示词的材质名交集。

口径按代码里真实用到的两级：
- **文件名精确命中**——`model/exemplars.py` 的 `by_mat` 就是按 `s["material"]`（文件名）建索引的，
  检索"同材质、其他画师"走的正是这一级；不足 2 张才退到 CLIP 文本近邻。
- **提示词命中**——去掉命名空间后的词串相同（`eval/prompts.py::prompt_words`），
  即模型看到的文本条件相同。

⚠ 另记一条代码事实（不是本脚本量的，是读代码读到的，写在这里免得下次重新发现）：
`eval/gen_trd.py:165` 的结构范例库**永远是 `load(16, ...)`**，`model/exemplars.py` 的
`draw()` 也固定返回 `[B,E,16,16]` —— 即**生成 32px 时喂进去的结构范例是 16px 瓦片**。
调色板记忆库（`model/palette_memory.py:66`，`sizes=(16,32)`）则是两种尺寸合在一起检索，
调色板与尺寸无关，所以调色板那一路**不存在 32px 饥饿**。

用法：python analysis/arch/pool32_coverage.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "eval"))
from prompts import prompt_words  # noqa: E402
from tiles_data import load  # noqa: E402

EXTRA = "train_extra_packs_only.json"     # v10/v11 系列训练时在用的那个


def pool_stats(size):
    rows = load(size, "train", extra=EXTRA)
    by_file, by_prompt = Counter(), Counter()
    packs_file, packs_prompt = {}, {}
    for s in rows:
        f = s["material"]
        p = " ".join(prompt_words(f))
        by_file[f] += 1
        by_prompt[p] += 1
        packs_file.setdefault(f, set()).add(s["pack"])
        packs_prompt.setdefault(p, set()).add(s["pack"])
    return rows, (by_file, packs_file), (by_prompt, packs_prompt)


def report(label, want, counts, packs):
    hit = [w for w in want if counts.get(w)]
    if not hit:
        print(f"    {label}：命中 0/{len(want)}")
        return
    tiles = sorted(counts[w] for w in hit)
    npk = [len(packs[w]) for w in hit]
    one_pack = sum(1 for c in npk if c == 1)
    print(f"    {label}：命中 {len(hit)}/{len(want)} = {len(hit) / len(want):.0%}"
          f"  样本数中位 {tiles[len(tiles) // 2]}（最多 {tiles[-1]}，只有 1 张的 {tiles.count(1)} 种）"
          f"  只出自 1 个包的 {one_pack} 种 = {one_pack / len(hit):.0%}")
    return set(hit)


def main():
    ent = json.loads((ROOT / "eval/prompt_sets.json").read_text(encoding="utf-8"))["E_mat"]
    want_file = sorted({e["material"] for e in ent})
    want_prompt = sorted({e["prompt"] for e in ent})
    print(f"测试集 E_mat：{len(ent)} 条，材质文件名 {len(want_file)} 种，提示词 {len(want_prompt)} 种")

    hits = {}
    for size in (16, 32):
        rows, (cf, pf), (cp, pp) = pool_stats(size)
        print(f"\n[{size}px 训练池 {EXTRA}] 瓦片 {len(rows)}  "
              f"包 {len({s['pack'] for s in rows})}  文件名 {len(cf)} 种")
        hits[size] = report("文件名精确命中", want_file, cf, pf)
        report("提示词命中    ", want_prompt, cp, pp)

    only16 = sorted(hits[16] - hits[32])
    print(f"\n**16px 池有、32px 池没有同名瓦片的测试材质：{len(only16)} 种**"
          f"（占 E_mat 材质的 {len(only16) / len(want_file):.0%}）")
    print("  前 30 例：" + ", ".join(w.replace(".png", "") for w in only16[:30]))


if __name__ == "__main__":
    main()
