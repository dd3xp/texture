"""同等采样预算下的重排：每材质 n 张里挑图文对齐分最高的一张（DALL·E 式 CLIP 重排）。

B2 本身是 best-of-4（按裁剪有效比挑），给 TRD 同样 4 张的预算是公平的。
**打分用 CLIP ViT-B/16（`weights/clip-vit-base-patch16`），评测用 B/32**——两个独立训练的模型，
避免拿评测模型本身来挑。结果存成 `<src>_rr<n>/<size>/<slug>_0.png`（每材质 1 张，与 B2 同口径）。

    python eval/rerank.py --src v2_retm4 --set V_mat --n 4
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from prompts import load_set     # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--set", default="V_mat")
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--clip", default=str(ROOT / "weights/clip-vit-base-patch16"))
    ap.add_argument("--root", type=Path, default=ROOT / "experiments/baselines")
    a = ap.parse_args()
    from transformers import CLIPModel, CLIPTokenizer
    dev = "cuda"
    m = CLIPModel.from_pretrained(a.clip).to(dev).eval()
    tok = CLIPTokenizer.from_pretrained(a.clip)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)
    prompts, _ = load_set(a.set)
    src = a.root / a.src / str(a.size)
    dst = a.root / f"{a.src}_rr{a.n}" / str(a.size)
    dst.mkdir(parents=True, exist_ok=True)
    picked = []
    for e in prompts:
        slug = e["material"].rsplit(".", 1)[0]
        tiles = [np.asarray(Image.open(src / f"{slug}_{k}.png").convert("RGB"))
                 for k in range(a.n) if (src / f"{slug}_{k}.png").exists()]
        if not tiles:
            continue
        x = torch.cat([F.interpolate(torch.from_numpy(t).permute(2, 0, 1)[None].float().to(dev), size=224,
                                     mode="nearest") for t in tiles]) / 255.0
        ie = F.normalize(m.get_image_features(pixel_values=(x - mean) / std).float(), dim=-1)
        t = tok([f"pixel art texture of {e['prompt']}"], return_tensors="pt").to(dev)
        te = F.normalize(m.get_text_features(**t).float(), dim=-1)
        best = int((ie @ te.T).squeeze(1).argmax())
        picked.append(best)
        Image.fromarray(tiles[best]).save(dst / f"{slug}_0.png")
    print(f"{len(picked)} 个材质；选中下标分布 {np.bincount(picked, minlength=a.n).tolist()} -> {dst}")


if __name__ == "__main__":
    main()
