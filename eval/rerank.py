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
def critic_pick(critic, cb, tiles, text):
    """每张瓦片 → (亮度序网格, 调色板码) → critic 的平均'被替换' logit，取最低。"""
    from tiles_data import canonicalise
    from train_trd import encode_palette
    from trd import K_MAX
    dev = text.device
    grids, pals, ks = [], [], []
    for t in tiles:
        cols, inv = np.unique(t.reshape(-1, 3), axis=0, return_inverse=True)
        g, p = canonicalise(inv.reshape(t.shape[:2]).astype(np.int64), cols.astype(np.uint8))
        p = p[:K_MAX]
        row = np.full(K_MAX, len(cb) + 1, np.int64)
        row[:len(p)] = encode_palette(p, cb)
        grids.append(np.minimum(g, len(p) - 1)); pals.append(row); ks.append(len(p))
    B = len(tiles)
    _, fake = critic(torch.tensor(np.stack(pals), device=dev), torch.tensor(np.stack(grids), device=dev),
                     torch.tensor(ks, device=dev), text[None].expand(B, -1), critic.null_color[None].expand(B, -1))
    return int(fake.float().mean((1, 2)).argmin())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--set", default="V_mat")
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--clip", default=str(ROOT / "weights/clip-vit-base-patch16"))
    ap.add_argument("--root", type=Path, default=ROOT / "experiments/baselines")
    ap.add_argument("--critic", type=Path, default=None,
                    help="改用 Token-Critic 打分（model/train_critic.py）：挑平均'被替换'概率最低的一张，不用任何 CLIP")
    ap.add_argument("--gen_run", type=Path, default=None, help="--critic 时：生成器 run（取码本）")
    a = ap.parse_args()
    from transformers import CLIPModel, CLIPTokenizer
    dev = "cuda"
    m = CLIPModel.from_pretrained(a.clip).to(dev).eval()
    tok = CLIPTokenizer.from_pretrained(a.clip)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)
    prompts, _ = load_set(a.set)
    critic = cb = te_b32 = None
    if a.critic is not None:                     # 判别网络打分：条件 = 文本（生成器同款 CLIP-B/32 文本嵌入）+ 瓦片自己的调色板
        sys.path.insert(0, str(ROOT / "model"))
        from train_critic import load_critic
        from train_trd import clip_text, TEXT_TMPL
        critic = load_critic(a.critic, dev)
        cb = np.load(a.gen_run / "codebook.npy")
        te_b32 = dict(zip([e["prompt"] for e in prompts],
                          clip_text([TEXT_TMPL.format(p=e["prompt"]) for e in prompts], dev).to(dev)))
    src = a.root / a.src / str(a.size)
    dst = a.root / (f"{a.src}_rr{a.n}" if a.critic is None else f"{a.src}_cr{a.n}") / str(a.size)
    dst.mkdir(parents=True, exist_ok=True)
    picked = []
    for e in prompts:
        slug = e["material"].rsplit(".", 1)[0]
        tiles = [np.asarray(Image.open(src / f"{slug}_{k}.png").convert("RGB"))
                 for k in range(a.n) if (src / f"{slug}_{k}.png").exists()]
        if not tiles:
            continue
        t = tok([f"pixel art texture of {e['prompt']}"], return_tensors="pt").to(dev)
        te = F.normalize(m.get_text_features(**t).float(), dim=-1)
        if critic is not None:
            best = critic_pick(critic, cb, tiles, te_b32[e["prompt"]])
        else:
            x = torch.cat([F.interpolate(torch.from_numpy(t).permute(2, 0, 1)[None].float().to(dev), size=224,
                                         mode="nearest") for t in tiles]) / 255.0
            ie = F.normalize(m.get_image_features(pixel_values=(x - mean) / std).float(), dim=-1)
            best = int((ie @ te.T).squeeze(1).argmax())
        picked.append(best)
        Image.fromarray(tiles[best]).save(dst / f"{slug}_0.png")
    print(f"{len(picked)} 个材质；选中下标分布 {np.bincount(picked, minlength=a.n).tolist()} -> {dst}")


if __name__ == "__main__":
    main()
