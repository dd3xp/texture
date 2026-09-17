"""(M53) 记账：各尺寸、各 split 的真人参照瓦片有多少张（零 GPU、零 API、纯计数）。

为什么问：(M53) 判 `VOID_FLOOR_TOO_NOISY` —— 32px 上 KID 的经验零分布极差 13.60，
比 (M46) 用的门槛 4.656 大 2.9 倍，而 S_gap 才 13.71 ⇒ 尺子比要量的东西还抖。
抖动的来源是参照集小（32px 只有 74 张，16px 有 196 张）。
本脚本只回答一个事实问题：**32px 的参照集在物理上最多能有多大**，
用来决定"把这把尺子修好"是否可能。⛔ 不产生任何判决、不解读 (M53) 的作废读数。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "eval"))
from tiles_data import load            # noqa: E402
from colour_task import targets, load_set, is_material   # noqa: E402


def main():
    print("=== 全部真人瓦片（按尺寸 × split）===")
    for size in (16, 24, 32):
        row = []
        for split in ("train", "val", "test"):
            try:
                n = len(load(size, split))
            except Exception as e:                      # noqa: BLE001
                n = f"ERR:{type(e).__name__}"
            row.append(f"{split}={n}")
        print(f"  size={size}  " + "  ".join(str(x) for x in row))

    print("\n=== 判官集合限定后的参照瓦片（= targets 实际拿到的）===")
    for set_name in ("V_mat", "E_mat"):
        for size in (16, 24, 32):
            try:
                T = targets(set_name, size)
                slugs = len({t["slug"] for t in T})
                print(f"  {set_name:6s} size={size}  tiles={len(T):4d}  slugs={slugs:3d}")
            except Exception as e:                      # noqa: BLE001
                print(f"  {set_name:6s} size={size}  ERR:{type(e).__name__}: {e}")

    print("\n=== val split 里、材质名通过 is_material 但不在 V_mat 提示表里的瓦片 ===")
    for size in (16, 24, 32):
        prompts, split = load_set("V_mat")
        have = {e["material"] for e in prompts}
        extra = [s for s in load(size, split)
                 if is_material(s["material"]) and s["material"] not in have]
        print(f"  size={size}  split={split}  未被 V_mat 覆盖的合格瓦片={len(extra)}"
              f"  材质={len({s['material'] for s in extra})}")


if __name__ == "__main__":
    main()
