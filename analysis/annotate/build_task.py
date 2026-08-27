"""生成人工标注任务：一个自包含的本地网页。

目的有两个，第二个可能更重要：
 1. 验证那把 12.8% 的分类器——它分高的图，人是不是真觉得更好
 2. 把负责人"效果并不好"的判断依据**变成可记录的数据**——
    他在看什么，目前只存在于他脑子里

三种条件两两配对：
    A 真人原生 16x16（留出作者，黄金标准）
    B 流水线，真人高分辨率源
    C 流水线，SDXL 源
配对 A-B / A-C / B-C，左右随机，材质名显示给标注者
（不告诉材质就没法判断"像不像木头"）。

另加注意力检查对：同一张图 vs 它被重度模糊的版本，答案显然。
标注者在这些对上选错，说明没认真看。

图片直接内嵌 16x16 原始 PNG（每张几百字节），
用 CSS image-rendering: pixelated 放大，所以整个文件很小且离线可用。
"""

import argparse
import base64
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "metric"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "premise"))
from compare import load_rgb, features                # noqa: E402
from build_testset import match_stats, palette_from   # noqa: E402
from materials import load as load_materials          # noqa: E402
from learn_material import HELD_OUT                   # noqa: E402

TEMPLATE = Path(__file__).with_name("task_template.html")


def b64(a: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(a.astype(np.uint8)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def tail(tex: Image.Image, n: int, ref: np.ndarray, pal: np.ndarray) -> np.ndarray:
    x = np.asarray(tex.resize((n, n), Image.BOX), float)
    x = match_stats(x, ref)
    d = ((x.reshape(-1, 1, 3) - pal.reshape(1, -1, 3)) ** 2).sum(-1)
    return pal[d.argmin(1)].reshape(x.shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=Path("data/contentdb/pairs.json"))
    ap.add_argument("--testset", type=Path, default=Path("experiments/metric/testset"))
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--n-materials", type=int, default=22)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("experiments/annotate/task.html"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    pairs = json.loads(args.pairs.read_text())
    names = load_materials()
    mats = [m for m in names if m in pairs][: args.n_materials]

    items = []
    for m in mats:
        held = [p for pk, p in pairs[m]["low"].items() if pk in HELD_OUT]
        if not held:
            continue
        A = load_rgb(held[0], args.size)
        ref = load_rgb(next(iter(pairs[m]["low"].values())), args.size)
        if A is None or ref is None:
            continue
        pal = palette_from(ref, features(ref)["n_colors"])

        hi = next(iter(pairs[m]["high"].values()))
        try:
            B = tail(Image.open(hi).convert("RGB"), args.size, ref, pal)
        except Exception:
            continue

        sd = args.testset / f"sdxl_{m}"
        if not sd.exists():
            sd = args.testset / f"sdxl_{m}.png"
        C = tail(Image.open(sd).convert("RGB"), args.size, ref, pal) if sd.exists() else None

        opts = {"A_artist_native": A, "B_pipe_artist": B}
        if C is not None:
            opts["C_pipe_sdxl"] = C
        keys = list(opts)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                l, r = (keys[i], keys[j]) if rng.random() < 0.5 else (keys[j], keys[i])
                items.append({"material": m, "label": names[m], "kind": "real",
                              "left": l, "right": r,
                              "limg": b64(opts[l]), "rimg": b64(opts[r])})

        # 注意力检查：原图 vs 重度模糊
        if rng.random() < 0.30:
            blur = np.stack([ndimage.gaussian_filter(A[..., c], 3.0) for c in range(3)], -1)
            l, r = ("good", "blur") if rng.random() < 0.5 else ("blur", "good")
            imgs = {"good": A, "blur": blur}
            items.append({"material": m, "label": names[m], "kind": "check",
                          "left": l, "right": r,
                          "limg": b64(imgs[l]), "rimg": b64(imgs[r])})

    rng.shuffle(items)
    n_check = sum(1 for i in items if i["kind"] == "check")
    print(f"生成 {len(items)} 对（含 {n_check} 对注意力检查）")

    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "__ITEMS__", json.dumps(items, ensure_ascii=False))
    args.out.write_text(html, encoding="utf-8")
    print(f"写入 {args.out}  ({args.out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
