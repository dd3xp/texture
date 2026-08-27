"""D1：把下载的材质包解压成带材质标签的瓦片数据集。

标签从文件名来——Luanti 生态里同一个方块在所有材质包里用同一个文件名
（`default_wood.png` 等），所以文件名天然是语义标签，不需要人工标注。

只收：
  - 方形图，边长属于 {16, 32, 64}。**不要求等于包的分辨率标签**——
    很多包内部是混合尺寸的，按标签卡会白扔掉大量可用瓦片
  - 通过材质过滤（排除工具、图标、植物、界面元素）
  - 不透明（像素画瓦片通常无 alpha；带 alpha 的多是物件而非平铺材质）

按**包**记录来源，D2 划分训练/验证/测试时要按包切，
不能按图随机切——同一个包的画风一致，随机切会泄漏。
"""

import argparse
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# 只保留可平铺的表面材质。这份名单沿用 analysis/metric/materials.py 的思路，
# 但放宽了词根限制——D1 要的是量，材质是否"典型"留到 D2 再筛。
DENY = re.compile(
    r"sapling|leaves|tool|sign|ladder|torch|_item|seed|bush|shrub|fern|papyrus|"
    r"lump|ingot|door|rail|chest|button|gui|hud|inventory|wield|hand|crack|"
    r"overlay|mask|particle|arrow|bubble|heart|crosshair|logo|menu|font|"
    r"_top$|_bottom$|_front$|_back$|_left$|_right$", re.I)


def unpack(manifest: dict, root: Path) -> None:
    for group, packs in manifest.items():
        for p in packs:
            dest = root / "unpacked" / group / f"{p['author']}__{p['name']}"
            if dest.exists():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(p["zip"]) as z:
                    z.extractall(dest)
            except Exception as e:
                print(f"  ! 解压失败 {p['name']}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/tiles_raw"))
    ap.add_argument("--out", type=Path, default=Path("data/tiles"))
    ap.add_argument("--min-packs", type=int, default=4,
                    help="一个材质至少要出现在这么多个包里才收")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.raw / "manifest.json").read_text())
    print("解压中…")
    unpack(manifest, args.raw)

    # 收集：material -> [(pack, group, size, path)]
    byMat = defaultdict(list)
    scanned = 0
    for group in sorted((args.raw / "unpacked").iterdir()):
        res = int(group.name.replace("res", ""))
        for pack in sorted(group.iterdir()):
            if not pack.is_dir():
                continue
            for f in pack.rglob("*.png"):
                scanned += 1
                if DENY.search(f.name):
                    continue
                try:
                    im = Image.open(f)
                except Exception:
                    continue
                w, h = im.size
                if w != h or w not in (16, 32, 64):
                    continue
                if im.mode in ("RGBA", "LA") or "transparency" in im.info:
                    a = np.asarray(im.convert("RGBA"))[..., 3]
                    if (a < 255).mean() > 0.02:   # 明显带透明 → 多半是物件
                        continue
                byMat[f.name].append(
                    {"pack": pack.name, "group": group.name, "size": w,
                     "path": str(f)})

    kept = {m: v for m, v in byMat.items()
            if len({x["pack"] for x in v}) >= args.min_packs}
    n_tiles = sum(len(v) for v in kept.values())
    packs = {x["pack"] for v in kept.values() for x in v}
    print(f"\n扫描 PNG {scanned}")
    print(f"材质（>= {args.min_packs} 个包覆盖）: {len(kept)}")
    print(f"瓦片总数: {n_tiles}")
    print(f"涉及包数: {len(packs)}")

    bysize = Counter(x["size"] for v in kept.values() for x in v)
    print(f"\n{'分辨率':<10}{'瓦片数':>10}")
    print("-" * 22)
    for s in sorted(bysize):
        print(f"{str(s)+'x'+str(s):<10}{bysize[s]:>10}")

    top = sorted(kept.items(), key=lambda kv: -len(kv[1]))[:12]
    print(f"\n{'覆盖最广的材质':<34}{'瓦片数':>8}{'包数':>7}")
    print("-" * 50)
    for m, v in top:
        print(f"{m[:33]:<34}{len(v):>8}{len({x['pack'] for x in v}):>7}")

    index = {"n_materials": len(kept), "n_tiles": n_tiles,
             "packs": sorted(packs), "by_size": dict(bysize),
             "materials": {m: v for m, v in kept.items()}}
    (args.out / "index.json").write_text(json.dumps(index, ensure_ascii=False))
    print(f"\n索引写入 {args.out/'index.json'}")


if __name__ == "__main__":
    main()
