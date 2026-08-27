"""D6：生成三方盲比任务（本模型 / 降采样基线 / 真人原生）。

设计取舍：

- **全材质都进，不挑对自己有利的样本。** D5 已经看出模型在几何结构材质上
  输给基线，但把那些排除掉就是挑样本，正是本项目要避免的错误。
- **不做"几何 vs 颗粒"二分类。** 试过用自相关峰强度分类，它能抓横条纹
  （wool / grass_side）却抓不到砖块的二维错缝（furnace_side、steel_block
  被误判为颗粒）。改为把结构分作为**连续变量**记录在每一对上，
  分析时看胜率随它怎么变，绕开分类不准的问题。
- **主对比是 模型 vs 基线**（论文要回答的问题），
  另加 模型 vs 真人、基线 vs 真人，用来定位两者离黄金标准多远。
- **样本量由功效分析定，不是拍脑袋。** 最初版只有 24 对主对比，
  模拟检验（`analyze_study.py --simulate`）显示它对"胜率随结构分
  0.75→0.25"这种强效应的功效只有 **31%**——即使效应真实存在也有
  69% 概率漏掉，那样标注就白花了。功效随对数：80 对 77%、120 对 92%。
  因此主对比改为**用满全部可比材质**，次对比只取子集。

问题措辞用"哪一张更像这个材质"——与之前那轮标注一致，可比。
"""

import argparse
import base64
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "metric"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model"))
from model import PixelTextureModel, generate      # noqa: E402
from build_testset import match_stats              # noqa: E402
from materials import prompt_for                   # noqa: E402


def b64(a: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(a.astype(np.uint8)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def struct_score(ix: np.ndarray, pal: np.ndarray) -> float:
    """1D 周期强度。抓横/竖条纹，抓不到二维错缝——所以只作连续协变量用。"""
    g = (pal.astype(float) @ np.array([0.299, 0.587, 0.114]))[ix]
    best = 0.0
    for ax in (0, 1):
        prof = g.mean(1 - ax)
        prof = prof - prof.mean()
        if prof.std() < 1e-9:
            continue
        ac = np.correlate(prof, prof, "full")[len(prof) - 1:]
        ac = ac / (ac[0] + 1e-12)
        if len(ac) > 2:
            best = max(best, float(ac[1:].max()))
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("experiments/model/reg/best.pt"))
    ap.add_argument("--data", type=Path, default=Path("data/tiles/dataset_k16.json"))
    ap.add_argument("--pairs", type=Path, default=Path("data/contentdb/pairs.json"))
    ap.add_argument("--n-materials", type=int, default=0,
                    help="0 表示用满全部可比材质（功效需要）")
    ap.add_argument("--n-secondary", type=int, default=30,
                    help="与真人对比的对数（次要问题，不需要同样功效）")
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("experiments/annotate/study.html"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    net = PixelTextureModel(k=ck["k"], n_materials=ck["n_materials"],
                            size=ck["size"], d=a["dim"], depth=a["depth"],
                            drop=0.0).to(dev).eval()
    net.load_state_dict(ck["state"])
    K, mat2id = ck["k"], ck["mat2id"]

    ds = json.loads(args.data.read_text())
    pairs = json.loads(args.pairs.read_text())
    ref = {}
    for s in ds["samples"]:
        if s["split"] == "test" and s["size"] == args.size and s["material"] not in ref:
            ref[s["material"]] = s
    mats = sorted(set(ref) & set(pairs) & set(mat2id))

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    # 按结构分排序后均匀取样，保证覆盖整个结构强度范围
    scored = []
    for m in mats:
        s = ref[m]
        ix = np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(args.size, args.size)
        scored.append((struct_score(ix, np.array(s["palette"], np.uint8)), m))
    scored.sort()
    if args.n_materials and args.n_materials < len(scored):
        step = max(1, len(scored) // args.n_materials)
        picked = [scored[i] for i in range(0, len(scored), step)][: args.n_materials]
    else:
        picked = scored
    print(f"取样 {len(picked)} 种材质，结构分 "
          f"{picked[0][0]:.3f} … {picked[-1][0]:.3f}")

    sec_on = set(rng.sample([m for _, m in picked],
                            min(args.n_secondary, len(picked))))
    items = []
    for score, m in picked:
        s = ref[m]
        pal = np.array(s["palette"], np.uint8)
        nk = len(pal)
        art = np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(
            args.size, args.size).astype(np.int64)
        art_rgb = pal[art]

        P = torch.zeros(1, K, 3, device=dev)
        P[0, :nk] = torch.tensor(pal.astype(np.float32) / 255.0, device=dev)
        V = torch.zeros(1, K, device=dev)
        V[0, :nk] = 1.0
        mid = torch.tensor([mat2id[m]], device=dev)
        gen = generate(net, P, V, mid, size=args.size, steps=args.steps,
                       device=dev)[0].cpu().numpy()
        model_rgb = pal[np.clip(gen, 0, nk - 1)]

        base_rgb = None
        for path in pairs[m]["high"].values():
            try:
                tex = Image.open(path).convert("RGB")
            except Exception:
                continue
            x = np.asarray(tex.resize((args.size,) * 2, Image.BOX), float)
            x = match_stats(x, art_rgb.astype(float))
            d = ((x.reshape(-1, 1, 3) - pal.astype(float).reshape(1, -1, 3)) ** 2).sum(-1)
            base_rgb = pal[d.argmin(1).reshape(args.size, args.size)]
            break
        if base_rgb is None:
            continue

        imgs = {"model": model_rgb, "baseline": base_rgb, "artist": art_rgb}
        duels = [("model", "baseline")]                 # 主对比，全部材质都做
        if len(sec_on) and m in sec_on:                 # 次对比只在子集上做
            duels.append(("model", "artist") if rng.random() < 0.5
                         else ("baseline", "artist"))
        for x, y in duels:
            l, r = (x, y) if rng.random() < 0.5 else (y, x)
            items.append({"material": m, "label": prompt_for(m), "kind": "real",
                          "struct": round(score, 4), "left": l, "right": r,
                          "limg": b64(imgs[l]), "rimg": b64(imgs[r])})

        if rng.random() < 0.05:                          # 注意力检查
            blur = np.stack([ndimage.gaussian_filter(art_rgb[..., c].astype(float), 3.0)
                             for c in range(3)], -1)
            g = {"good": art_rgb, "blur": blur}
            l, r = ("good", "blur") if rng.random() < 0.5 else ("blur", "good")
            items.append({"material": m, "label": prompt_for(m), "kind": "check",
                          "struct": round(score, 4), "left": l, "right": r,
                          "limg": b64(g[l]), "rimg": b64(g[r])})

    rng.shuffle(items)
    n_check = sum(1 for i in items if i["kind"] == "check")
    print(f"生成 {len(items)} 对（其中注意力检查 {n_check} 对）")

    tpl = (Path(__file__).parent / "task_template.html").read_text(encoding="utf-8")
    args.out.write_text(tpl.replace("__ITEMS__", json.dumps(items, ensure_ascii=False)),
                        encoding="utf-8")
    print(f"写入 {args.out}  ({args.out.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
