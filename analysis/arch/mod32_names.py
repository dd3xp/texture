"""32px 模组瓦片的名字盘点（零 GPU、零 API，只读已有 JSON）——**为设计筛选判据服务**。

背景：`data32_supply.py`（`e9531a0`）盘出含模组的 32px 池是 2590 张 / 170 包，比在用的
641 张 / 24 包多 4 倍、146 个全新来源。但账本同时记着：按张数排头的模组来源全是**非平铺的东西**
（字母表、音标、机器面板、人物立绘），整包倒进来在 16px 上被否过（2026-09-11，KID 12.8 -> 16.3）。

所以下一题不是"要不要用模组数据"，而是"**能不能只挑出其中可平铺的材质**"。
本脚本**只列名字与来源分布**，不做任何筛选判断、不看任何胜负数字，
目的是让筛选判据能照着真实的名字写，而不是拍脑袋。

⚠ 本脚本不是筛子，也不产出训练文件。筛选判据另行预注册。

用法（服务器上跑，需要 data/tiles/train_extra.json）：
    python analysis/arch/mod32_names.py [--top 60]
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "eval"))
from prompts import prompt_words  # noqa: E402
from tiles_data import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=60)
    args = ap.parse_args()

    packs_only = {(s["pack"], s["material"])
                  for s in load(32, "train", extra="train_extra_packs_only.json")}
    rows = load(32, "train", extra="train_extra.json")
    new = [s for s in rows if (s["pack"], s["material"]) not in packs_only]

    print(f"含模组 32px 共 {len(rows)} 张；其中在用池里已有 {len(rows) - len(new)} 张，"
          f"**新增 {len(new)} 张**")

    src = Counter(s["pack"] for s in new)
    print(f"\n新增来源 {len(src)} 个，按张数前 {args.top}：")
    for p, c in src.most_common(args.top):
        names = sorted({s["material"] for s in new if s["pack"] == p})
        print(f"  {c:4d}  {p}")
        print(f"        {', '.join(names[:12])}{' ...' if len(names) > 12 else ''}")

    print(f"\n新增材质名共 {len({s['material'] for s in new})} 种，按出现次数前 {args.top}：")
    for m, c in Counter(s["material"] for s in new).most_common(args.top):
        print(f"  {c:4d}  {m}")

    # ---- 候选筛子的产量（只是算给人看，**不是判据、也不产出训练文件**）----
    # 词表只从**训练包**的材质名取：拿测试材质名当筛子＝选数据时偷看测试集，绝对不许。
    vocab_file, vocab_word = set(), set()
    for n in (16, 32, 64):
        for s in load(n, "train", extra="train_extra_packs_only.json"):
            vocab_file.add(s["material"])
            vocab_word.update(prompt_words(s["material"]))
    for tag, keep in (("文件名同名", [s for s in new if s["material"] in vocab_file]),
                      ("词全在训练包词表内",
                       [s for s in new if prompt_words(s["material"])
                        and set(prompt_words(s["material"])) <= vocab_word])):
        src2 = Counter(s["pack"] for s in keep)
        h = hhi(src2) if keep else 1.0
        print(f"\n[候选筛子：{tag}] 留下 {len(keep)} 张 / {len(src2)} 个来源"
              f"（有效包数 1/HHI = {1 / h:.1f}）")
        for p, c in src2.most_common(12):
            print(f"    {c:4d}  {p}")
        print("    材质名样例：" + ", ".join(sorted({s['material'] for s in keep})[:20]))


def hhi(counts):
    n = sum(counts.values())
    return sum((c / n) ** 2 for c in counts.values()) if n else 1.0


if __name__ == "__main__":
    main()
