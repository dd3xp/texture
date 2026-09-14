"""32px 训练数据的供给盘点（零 GPU、零 API，只读已有 JSON）。

**为什么问这个**：32px 是唯一还在的缺口（判官判 TRD 输 B2，41%，p=0.014）。推理侧八条杠杆
全否，训练侧"在现有池子里调采样权重"（pack_balance γ=0.5）也打平（`11572ef`）。账本因此写着：
γ=1 若也否，下一轮的题只能是**换掉数据本身**。换之前先盘点——**有没有更多 32px 真人数据**，
以及它会不会只是把同样那几个包再数一遍。

三个池子（都过 `tiles_data.load` 的去污染，即训练时实际看到的那批）：
- `train_extra_packs_only.json`：v10 / v11 系列在用的（`runs/trd_v10/last.pt` 的 `extra_file`）。
- `train_extra.json`：同一套构建规则，但**含 Minetest 模组贴图**（`build_extra_train.py --mods`）。
- `train_64to32.json`：64px 按 2×2 众数降下来的 1236 张，v11d 用过。

⚠ **这不是"选池子"的判据，本脚本一个胜负数字都不看**。模组数据在 16px 上是**试过并被否掉**的
（2026-09-11：v5/v6 含模组 KID 16.3，干净材质包 v4 是 12.8，采样明显变差）——所以这里只回答
"供给有多少、来自哪里、是不是新包"，要不要用得另开预注册的臂去判。

用法：python analysis/arch/data32_supply.py
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from tiles_data import load  # noqa: E402

POOLS = [
    ("v10 在用（材质包）", "train_extra_packs_only.json"),
    ("含模组", "train_extra.json"),
    ("v11d 用过的 64->32", "train_extra_packs_only.json+train_64to32.json"),
]


def hhi(counts):
    """Herfindahl 指数：Σ 份额²。1 = 全在一个包里，1/n = 完全均匀。"""
    n = sum(counts.values())
    return sum((c / n) ** 2 for c in counts.values())


def profile(name, rows):
    packs = Counter(r["pack"] for r in rows)
    top = packs.most_common(5)
    share = sum(c for _, c in top) / len(rows)
    print(f"\n[{name}] 瓦片 {len(rows)}  包 {len(packs)}  HHI {hhi(packs):.3f}  "
          f"前五包占 {share:.0%}  有效包数 1/HHI = {1 / hhi(packs):.1f}")
    for p, c in top:
        print(f"    {c:5d}  {p}")
    return packs


def main():
    pools = {}
    for name, fn in POOLS:
        pools[fn] = load(32, "train", extra=fn)
        profile(name, pools[fn])

    base = pools["train_extra_packs_only.json"]
    full = pools["train_extra.json"]
    base_packs = {r["pack"] for r in base}
    base_keys = {(r["pack"], r["material"]) for r in base}
    new = [r for r in full if (r["pack"], r["material"]) not in base_keys]
    new_packs = Counter(r["pack"] for r in new)

    print(f"\n=== 模组能补上来的增量 ===")
    print(f"新增瓦片 {len(new)}（池子 {len(base)} -> {len(full)}，{len(full) / len(base):.1f} 倍）")
    only_new = [p for p in new_packs if p not in base_packs]
    print(f"新增来源 {len(new_packs)} 个")
    print(f"全新来源 {len(only_new)} 个，贡献 {sum(new_packs[p] for p in only_new)} 张")
    print("  新来源里最大的二十个（⚠ 看名字：字母表 / 音标 / 机器面板 / 人物立绘"
          " 不是可平铺的材质，这正是 2026-09-11 模组数据在 16px 上让采样变差的那批）：")
    for p, c in sorted(((p, new_packs[p]) for p in only_new), key=lambda kv: -kv[1])[:20]:
        print(f"    {c:5d}  {p}")

    # v11d 那 1236 张到底是什么：按包看，不是"64->32 降采样"这件事本身，而是来源集中度
    d32 = pools["train_extra_packs_only.json+train_64to32.json"]
    add = Counter(r["pack"] for r in d32)
    add.subtract(Counter(r["pack"] for r in base))
    add = Counter({p: c for p, c in add.items() if c > 0})
    big, bign = add.most_common(1)[0]
    print(f"\n=== v11d 那批 64->32 的来源 ===")
    print(f"共 {sum(add.values())} 张，来自 {len(add)} 个包；最大一个 {big} 占 "
          f"{bign}/{sum(add.values())} = {bign / sum(add.values()):.0%}")

    # 增量是不是"同一张图换个包名"：结构键（亮度序索引网格）去重后还剩多少
    def key(r):
        return (r["idx"].tobytes(), r["idx"].shape[0])

    base_struct = {key(r) for r in base}
    new_struct = {key(r) for r in new}
    print(f"\n结构去重：现有池 {len(base_struct)} 种 / {len(base)} 张；"
          f"增量 {len(new_struct)} 种 / {len(new)} 张；"
          f"增量中与现有池结构相同的 {len(new_struct & base_struct)} 种")
    print(f"并集结构种数 {len(base_struct | new_struct)}"
          f"（相对现有池 {len(base_struct | new_struct) / len(base_struct):.1f} 倍）")


if __name__ == "__main__":
    main()
