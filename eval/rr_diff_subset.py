"""重排真正换了样本的那些材质 -> 判官 subset。

`eval/rerank.py --n 4` 在 4 张里按 CLIP B/16 挑一张；挑中第 0 张时，重排后的图与"单张"
**逐像素相同**，判官拿到两张一样的图只能随机作答或两序不一致，白白稀释效应。
这里把"挑中的不是第 0 张"的材质挑出来写成 `judge_pairs.py --subset` 认的格式。

筛选只看 CLIP 分数（挑样本那一步本来就只看它），**与判官胜负无关** -> 不是按结果选样本。

    python eval/rr_diff_subset.py --src nf8_name --rr nf8_name_rr4 --set V_mat --out eval/rr4_diff.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from prompts import load_set     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--rr", required=True)
    ap.add_argument("--set", default="V_mat")
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--root", type=Path, default=ROOT / "experiments/baselines")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    items, same, missing = [], 0, 0
    for e in load_set(a.set)[0]:
        slug = e["material"].rsplit(".", 1)[0]
        pa = a.root / a.src / str(a.size) / f"{slug}_0.png"
        pb = a.root / a.rr / str(a.size) / f"{slug}_0.png"
        if not pa.exists() or not pb.exists():
            missing += 1
            continue
        x, y = (np.asarray(Image.open(p).convert("RGB")) for p in (pa, pb))
        if x.shape == y.shape and (x == y).all():
            same += 1
        else:
            items.append(e)
    a.out.write_text(json.dumps({"items": items}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"重排换了样本的材质 {len(items)} 个；逐像素相同 {same} 个（挑中第 0 张）；缺图 {missing} -> {a.out}")


if __name__ == "__main__":
    main()
