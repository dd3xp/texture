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

用法：python analysis/arch/pool32_coverage.py            # 默认 E_mat + 旧 extra = `6e2a95f` 那份逐字复现
     python analysis/arch/pool32_coverage.py --set V_mat --extra train_extra_packs_only.json+train_64to32.json

---------------------------------------------------------------------------------------------------
**2026-09-15 追加：`--p_tile16` 那轮的操作检验（判据跑前写死，见下）**

`scripts/trd_tile16_train_eval.sh` 的前提 2 引的是**上面这份 E_mat 数字**（32px 覆盖 36%、16px 覆盖 68%），
可那一轮的指标判据 ①②③ 全部在 **V_mat（验证集）** 上量。两个集合的覆盖率没有理由相同 →
**干预在被测集上到底有没有覆盖空间，从来没量过**。不先量，将来 ② 判"什么也没测到"时就分不开：
(a) 假设错；(b) 被测集上压根没有可补的覆盖（＝功效不足）。本脚本一个胜负数字都不看，只数材质名交集。

判据（**在跑之前写死**，`--extra` 取训练实际用的 `train_extra_packs_only.json+train_64to32.json`，
口径取"文件名精确命中"＝`model/exemplars.py::by_mat` 真正用的那一级）：

- headroom := 16px 池命中率 − 32px 池命中率（单位 pp），在 **V_mat** 上量。
- **通过**：headroom ≥ 16pp（＝当初立题的 E_mat 那份 32pp 的一半）→ 干预在被测集上确有覆盖空间，
  ② 按 `trd_tile16_train_eval.sh` 里写的读。
- **不通过**：headroom < 16pp → ⛔ ② 的空结果**只能读作"功效不足"**，不许读作"假设被否"；
  也不许因此放宽或改写 ②。
- ⚠ 本检验**只管功效前提，不改任何判据、不下任何假设层面的判决**；通过与否都不授权新臂。
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "eval"))
from prompts import load_set, prompt_words  # noqa: E402
from tiles_data import load  # noqa: E402

# ⚠ 这是 **v10** 训练时用的那个（`runs/trd_v10/last.pt` 的 ckpt["args"]["extra_file"] 确认）。
# **v11d 及其后（含 --p_tile16 那轮）用的是 `train_extra_packs_only.json+train_64to32.json`**，
# 32px 池因此从 641 张/24 包涨到 1877 张/28 包 —— `6e2a95f` 那份 36% 只对 v10 的池子成立，
# 换成 v11d 真正的池子是 62%。默认值保持不变只为逐字复现 `6e2a95f`，**别当成"训练在用的"**。
EXTRA = "train_extra_packs_only.json"


def pool_stats(size, extra):
    rows = load(size, "train", extra=extra)
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
        return set()
    tiles = sorted(counts[w] for w in hit)
    npk = [len(packs[w]) for w in hit]
    one_pack = sum(1 for c in npk if c == 1)
    print(f"    {label}：命中 {len(hit)}/{len(want)} = {len(hit) / len(want):.0%}"
          f"  样本数中位 {tiles[len(tiles) // 2]}（最多 {tiles[-1]}，只有 1 张的 {tiles.count(1)} 种）"
          f"  只出自 1 个包的 {one_pack} 种 = {one_pack / len(hit):.0%}")
    return set(hit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="E_mat", help="E_mat（默认，= `6e2a95f` 那份）或 V_mat（--p_tile16 那轮真正在量的集合）")
    ap.add_argument("--extra", default=EXTRA, help="训练实际用的 extra 文件，允许 'a.json+b.json'")
    a = ap.parse_args()

    ent, _ = load_set(a.set)
    want_file = sorted({e["material"] for e in ent})
    want_prompt = sorted({e["prompt"] for e in ent})
    print(f"{a.set}：{len(ent)} 条，材质文件名 {len(want_file)} 种，提示词 {len(want_prompt)} 种")

    hits, rate = {}, {}
    for size in (16, 32):
        rows, (cf, pf), (cp, pp) = pool_stats(size, a.extra)
        print(f"\n[{size}px 训练池 {a.extra}] 瓦片 {len(rows)}  "
              f"包 {len({s['pack'] for s in rows})}  文件名 {len(cf)} 种")
        hits[size] = report("文件名精确命中", want_file, cf, pf)
        report("提示词命中    ", want_prompt, cp, pp)
        rate[size] = len(hits[size]) / len(want_file)

    only16 = sorted(hits[16] - hits[32])
    print(f"\n**16px 池有、32px 池没有同名瓦片的 {a.set} 材质：{len(only16)} 种**"
          f"（占 {a.set} 材质的 {len(only16) / len(want_file):.0%}）")
    print("  前 30 例：" + ", ".join(w.replace(".png", "") for w in only16[:30]))

    # ---- `--p_tile16` 那轮的操作检验：干预在**被测集**上有没有覆盖空间（判据见文件头，跑前写死）----
    head = 100 * (rate[16] - rate[32])
    mixed = len(hits[16] | hits[32]) / len(want_file)
    print(f"\n[操作检验 --p_tile16] 32px 池 {rate[32]:.1%}  16px 池 {rate[16]:.1%}  "
          f"headroom = {head:.1f}pp；p_tile16>0 时 32px 那一路能见到的并集 = {mixed:.1%}")
    print(f"  判据：headroom ≥ 16pp 为通过 → **{'通过' if head >= 16 else '不通过'}**")
    if head < 16:
        print("  ⛔ 不通过 → `trd_tile16_train_eval.sh` 判据②的空结果只能读作「功效不足」，"
              "不许读作「假设被否」，也不许因此改写②。")


if __name__ == "__main__":
    main()
