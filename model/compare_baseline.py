"""D5：本模型 vs 降采样基线 vs 真人原生，同材质同调色板同区域。

三组用**完全相同的调色板**（取自该材质在测试划分里的真人瓦片），
所以调色板不构成组间差异。

指标：
  1. 调色板合规率 —— 三组都应 100%（基线也量化到同一调色板），
     列出来是为了证明比较是公平的，不是我们占了便宜
  2. 区域 IoU —— 模型带 mask 生成，检验区域外是否保持背景
  3. 跳档:相邻 比值 —— 技术路线的核心主张。
     真人 12:1、降采样 4:1（来自 analysis/structure_grain）。
     模型若落在真人一侧，说明"在索引空间直接生成"确实避开了平均的宿命。

**关于这个比值的保留**：它来自结构/噪点第一版，用 3×3 众数当"结构"，
分不开结构边缘与噪点，所以**不能解读成"噪点占比"**。
但它确实能区分真人与降采样，作为判别性统计量可用，
且三组用同一把尺子，结论是相对的。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "metric"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "premise"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "structure_grain"))
from model import build_model, generate            # noqa: E402
from build_testset import match_stats              # noqa: E402
from decompose import grain_stats                  # noqa: E402


def idx_to_rgb(idx: np.ndarray, pal: np.ndarray) -> np.ndarray:
    return pal[np.clip(idx, 0, len(pal) - 1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("experiments/model/reg/best.pt"))
    ap.add_argument("--data", type=Path, default=Path("data/tiles/dataset_k16.json"))
    ap.add_argument("--pairs", type=Path, default=Path("data/contentdb/pairs.json"))
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("experiments/compare"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    arch = ck.get("arch", "transformer")   # 旧 checkpoint 没有这个字段
    net = build_model(arch, k=ck["k"], n_materials=ck["n_materials"],
                      size=ck["size"], d=a["dim"], depth=a["depth"],
                      drop=0.0).to(dev).eval()
    print(f"模型架构 {arch}  d={a['dim']} depth={a['depth']}")
    net.load_state_dict(ck["state"])
    mat2id = ck["mat2id"]
    K = ck["k"]

    ds = json.loads(args.data.read_text())
    pairs = json.loads(args.pairs.read_text())
    # 每个材质取一张测试划分（未见作者）的真人瓦片作参照与调色板来源
    ref = {}
    for s in ds["samples"]:
        if s["split"] == "test" and s["size"] == args.size and s["material"] not in ref:
            ref[s["material"]] = s
    mats = sorted(set(ref) & set(pairs) & set(mat2id))
    print(f"三方可比材质 {len(mats)} 种\n")

    rows, gallery = [], []
    for m in mats:
        s = ref[m]
        pal = np.array(s["palette"], np.float32)
        nk = len(pal)
        art_idx = np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(
            args.size, args.size).astype(np.int64)

        palt = torch.zeros(1, K, 3, device=dev)
        palt[0, :nk] = torch.tensor(pal / 255.0, device=dev)
        valt = torch.zeros(1, K, device=dev)
        valt[0, :nk] = 1.0
        mid = torch.tensor([mat2id[m]], device=dev)

        gen = generate(net, palt, valt, mid, size=args.size,
                       steps=args.steps, device=dev)[0].cpu().numpy()

        # 基线：高分辨率源 → 色彩对齐 → 量化到同一调色板
        art_rgb = idx_to_rgb(art_idx, pal)
        base_idx = None
        for path in pairs[m]["high"].values():
            try:
                tex = Image.open(path).convert("RGB")
            except Exception:
                continue
            x = np.asarray(tex.resize((args.size,) * 2, Image.BOX), float)
            x = match_stats(x, art_rgb.astype(float))
            d = ((x.reshape(-1, 1, 3) - pal.reshape(1, -1, 3)) ** 2).sum(-1)
            base_idx = d.argmin(1).reshape(args.size, args.size)
            break
        if base_idx is None:
            continue

        # 区域约束：给模型一个圆形 mask，检查区域外是否保持 BG
        yy, xx = np.mgrid[:args.size, :args.size] - (args.size - 1) / 2
        reg = torch.tensor((yy ** 2 + xx ** 2) <= (args.size * 0.35) ** 2,
                           device=dev)[None]
        gen_r = generate(net, palt, valt, mid, size=args.size, steps=args.steps,
                         region=reg, device=dev)[0].cpu().numpy()
        rmask = reg[0].cpu().numpy()
        iou_ok = bool((gen_r[~rmask] == net.BG).all())
        inside_ok = bool(((gen_r[rmask] >= 0) & (gen_r[rmask] < nk)).all())

        rec = {"material": m, "n_colors": nk,
               "region_outside_bg": iou_ok, "region_inside_valid": inside_ok}
        for tag, ix in (("artist", art_idx), ("baseline", base_idx), ("model", gen)):
            rgb = idx_to_rgb(ix, pal)
            g = grain_stats(rgb.astype(np.float64), pal.astype(np.float64))
            ratio = g["frac_ge2"] / max(g["frac_pm1"], 1e-9)
            rec[f"{tag}_ratio"] = ratio
            rec[f"{tag}_flat"] = g["frac_zero"]
            rec[f"{tag}_in_palette"] = bool(((ix >= 0) & (ix < nk)).all())
        rows.append(rec)
        gallery.append((m, pal, art_idx, base_idx, gen))

    print(f"{'组':<12}{'调色板合规':>12}{'纯结构占比':>12}{'跳档:相邻 中位':>16}")
    print("-" * 54)
    for tag, label in (("artist", "真人原生"), ("baseline", "降采样基线"), ("model", "本模型")):
        ok = np.mean([r[f"{tag}_in_palette"] for r in rows])
        flat = np.median([r[f"{tag}_flat"] for r in rows])
        ratio = np.median([r[f"{tag}_ratio"] for r in rows])
        print(f"{label:<12}{ok:>11.0%}{flat:>12.3f}{ratio:>16.2f}")

    print(f"\n区域约束（{len(rows)} 个材质，圆形 mask）：")
    print(f"  区域外全部保持背景: {np.mean([r['region_outside_bg'] for r in rows]):.0%}")
    print(f"  区域内全部合法索引: {np.mean([r['region_inside_valid'] for r in rows]):.0%}")

    (args.out / "metrics.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))

    # 画廊图
    import random
    random.seed(0)
    pick = random.sample(gallery, min(14, len(gallery)))
    CELL, PAD, LAB, HDR = 96, 4, 150, 26
    W = LAB + 3 * (CELL + PAD) + PAD
    H = HDR + len(pick) * (CELL + PAD) + PAD
    from PIL import ImageDraw
    cv = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(cv)
    for ci, t in enumerate(["真人原生", "降采样基线", "本模型"]):
        dr.text((LAB + ci * (CELL + PAD) + 4, 6), t, fill="black")
    for ri, (m, pal, ai, bi, gi) in enumerate(pick):
        y = HDR + ri * (CELL + PAD)
        dr.text((6, y + CELL // 2 - 5),
                m.replace("default_", "").replace(".png", "")[:20], fill="black")
        for ci, ix in enumerate([ai, bi, gi]):
            x = LAB + ci * (CELL + PAD)
            im = Image.fromarray(idx_to_rgb(ix, pal).astype(np.uint8))
            cv.paste(im.resize((CELL, CELL), Image.NEAREST), (x, y))
            dr.rectangle([x, y, x + CELL, y + CELL], outline="#bbb")
    cv.save(args.out / "gallery.png")
    print(f"\n画廊写入 {args.out/'gallery.png'}")


if __name__ == "__main__":
    main()
