"""补训练数据：`dataset_k16.json` 丢掉的训练包瓦片（**只加训练集，val/test 一张不动**）。

原数据集（`analysis/dataset/build_tiles.py`）只收"至少出现在 4 个包里"的材质——那是为跨画师
对齐做研究用的；对生成模型来说，训练包里只出现 1–3 次的材质也是真人画的有效样本，被白扔了。
v2 在第 6000 步后过拟合（16px 训练集只有 3395 张），数据是瓶颈。

规则（其余与原数据集完全一致：同一 DENY 名单、方形 16/32/64、不透明、原生色 ≥3、中位切分 K=16）：
- 只收**训练包**：dataset_k16 里属于 train 的包；原数据集里一张都没进的包也算训练包，
  **除非**它与任何 val/test 包同作者（按包名 `作者__名字` 的作者段判断，防画风泄漏）。
- 去掉 dataset_k16 里已有的（同包、同材质、同尺寸）。
- **去重防泄漏**：量化后与任何 val/test 瓦片逐像素相同的，丢掉（材质包常原样拷贝上游贴图）。

输出 `data/tiles/train_extra.json`，样本格式与 dataset_k16 相同，另加 `"extra": true`。
用法（服务器上跑，原始包在 data/tiles_raw/unpacked）：
    python model/build_extra_train.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis" / "dataset"))
from build_tiles import DENY          # noqa: E402  与原数据集同一份名单
from prepare import MIN_COLORS, quantize   # noqa: E402


def tile_key(idx: np.ndarray, pal: np.ndarray) -> bytes:
    return np.asarray(pal, np.uint8)[idx].tobytes()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mods", type=Path, default=None, help="fetch_mod_tiles.py 的输出目录（可选）")
    args = ap.parse_args()
    ds = json.loads((ROOT / "data/tiles/dataset_k16.json").read_text())
    samples = ds["samples"]
    split_of = {s["pack"]: s["split"] for s in samples}
    held_authors = {p.split("__")[0].lower() for p, sp in split_of.items() if sp != "train"}
    have = {(s["pack"], s["material"], s["size"]) for s in samples}
    held_keys = set()
    for s in samples:
        if s["split"] != "train":
            n = s["size"]
            idx = np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(n, n)
            held_keys.add(tile_key(idx, s["palette"]))

    out, drop = [], Counter()
    raw = ROOT / "data/tiles_raw/unpacked"
    dirs = [(pk, pk.name) for group in sorted(raw.iterdir()) for pk in sorted(group.iterdir())]
    if args.mods:                                   # fetch_mod_tiles.py 的输出：<作者>__<模组名>/*.png
        dirs += [(m, m.name + "@mod") for m in sorted(args.mods.iterdir())]
    for pack, pname in dirs:
        if not pack.is_dir():
            continue
        sp = split_of.get(pname)
        if sp is None and pname.split("__")[0].lower() in held_authors:
            drop["同作者包"] += 1
            continue
        if sp not in (None, "train"):
            continue
        seen = set()
        for f in sorted(pack.rglob("*.png")):
            if DENY.search(f.name):
                continue
            try:
                im = Image.open(f)
                w, h = im.size
            except Exception:
                continue
            if w != h or w not in (16, 32, 64):
                continue
            if (pname, f.name, w) in have or (f.name, w) in seen:
                drop["已有/包内重名"] += 1
                continue
            if im.mode in ("RGBA", "LA") or "transparency" in im.info:
                if (np.asarray(im.convert("RGBA"))[..., 3] < 255).mean() > 0.02:
                    drop["透明"] += 1
                    continue
            a = np.asarray(im.convert("RGB"))
            native = len(np.unique(a.reshape(-1, 3), axis=0))
            if native < MIN_COLORS:
                drop["少于3色"] += 1
                continue
            ind, pal = quantize(a, 16)
            if tile_key(ind, pal) in held_keys:
                drop["与val/test逐像素相同"] += 1
                continue
            seen.add((f.name, w))
            out.append({"material": f.name, "pack": pname, "size": w, "split": "train",
                        "native_colors": native, "k_used": int(len(pal)),
                        "idx": ind.astype(np.uint8).tobytes().hex(), "palette": pal.tolist(),
                        "extra": True})
    by = Counter(s["size"] for s in out)
    print(f"新增训练瓦片 {len(out)}（" + "  ".join(f"{k}px:{v}" for k, v in sorted(by.items())) + "）")
    print(f"新增包 {len({s['pack'] for s in out} - set(split_of))}（其中模组 "
          f"{len({s['pack'] for s in out if s['pack'].endswith('@mod')})}），材质 {len({s['material'] for s in out})}")
    print("丢弃:", dict(drop))
    p = ROOT / "data/tiles/train_extra.json"
    p.write_text(json.dumps({"k": 16, "samples": out}))
    print("->", p)


if __name__ == "__main__":
    main()
