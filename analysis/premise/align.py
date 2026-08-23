"""解压材质包并按文件名对齐，找出两组共有的材质。

Luanti 的材质包沿用同一套文件名（default_stone.png 等），所以文件名
本身就是语义对齐的锚点，不需要任何标注。
"""

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image


def unpack(manifest: dict, root: Path) -> None:
    for group, packs in manifest.items():
        for p in packs:
            dest = root / group / f"{p['author']}__{p['name']}"
            if dest.exists():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(p["zip"]) as z:
                    z.extractall(dest)
            except Exception as e:
                print(f"  ! 解压失败 {p['name']}: {e}")


def index(root: Path, group: str, want) -> dict:
    """basename -> {pack: path}，只收方形且尺寸符合 want() 的图。"""
    idx = defaultdict(dict)
    gdir = root / group
    if not gdir.exists():
        return idx
    for pack in sorted(gdir.iterdir()):
        if not pack.is_dir():
            continue
        for f in pack.rglob("*.png"):
            try:
                w, h = Image.open(f).size
            except Exception:
                continue
            if w != h or not want(w):
                continue
            # 同名文件在一个包里可能出现多次，保留第一个
            idx[f.name].setdefault(pack.name, str(f))
    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/contentdb"))
    ap.add_argument("--low-size", type=int, default=16)
    ap.add_argument("--high-min", type=int, default=64)
    ap.add_argument("--min-packs", type=int, default=3,
                    help="一个材质至少要在两组各出现这么多个包里才算数")
    args = ap.parse_args()

    manifest = json.loads((args.root / "manifest.json").read_text())
    unpack(manifest, args.root)

    low = index(args.root, "low", lambda s: s == args.low_size)
    high = index(args.root, "high", lambda s: s >= args.high_min)
    print(f"low  组: {len(low)} 个不同文件名")
    print(f"high 组: {len(high)} 个不同文件名")

    pairs = {}
    for name in sorted(set(low) & set(high)):
        nl, nh = len(low[name]), len(high[name])
        if nl >= args.min_packs and nh >= args.min_packs:
            pairs[name] = {"low": low[name], "high": high[name]}

    print(f"\n两组各 >={args.min_packs} 个包覆盖的材质: {len(pairs)}")
    for name, v in sorted(pairs.items(), key=lambda kv: -(len(kv[1]['low']) + len(kv[1]['high'])))[:30]:
        print(f"  {name[:44]:<44} low={len(v['low']):>2} high={len(v['high']):>2}")

    out = args.root / "pairs.json"
    out.write_text(json.dumps(pairs, indent=1))
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
