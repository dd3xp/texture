"""为 TRD v3 准备"大模型参考图"条件：每个材质用 SDXL 渲若干张 1024 图，存其 CLIP 图像嵌入。

动机（`docs/arch_plan.md` §1 条件 2；v2 的短板是 CLIP 文本对齐最低）：
大模型知道"这个材质长什么样"，但不知道像素画家怎么画；TRD 从 4951 张真人瓦片学到画法，
但开放语义学不全。把 SDXL 渲染图的**全局语义嵌入**（不是像素）喂给 TRD，
让它学"参考图语义 → 真人画法"的映射——而不是像 B1/B2 那样直接降采样大模型的像素。

覆盖：train[+extra] + val + test（训练要用到的全部材质，见 all_prompts 的注释）
+ E-mat（评测时推理用，属于方法的推理成本，与 B1 同等）。
提示词模板与 B1 相同。种子 = seed + 材质序号 + 1000*样本号。
输出 `experiments/refs/emb_<name>.pt`：{prompt -> [K, 512] 归一化 CLIP-B/32 图像嵌入}，
外加每个提示词第 0 张的 128px 缩略图（便于目视核对）。可断点续跑。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "eval"))
from tiles_data import load                  # noqa: E402
from prompts import prompt_words, load_set   # noqa: E402

TMPL = ("pixel art, {p}, top-down seamless tileable game texture, "
        "flat lighting, no shadows, orthographic, chunky large pixels")
NEG = "perspective, 3d render, vignette, watermark, text, border, blurry"


def all_prompts(extra=False):
    """覆盖口径必须与 `train_trd.py` 建 `mats` 的口径**一模一样**（train[+extra] / val / test，16 与 32），
    再并上 E_mat 的提示词（推理期要用）。少一个提示词，训练就直接死在 `train_trd.py:383`
    的"参考图缺材质"检查上；(M55) 补 `extra` 与 `test` 正是为此。"""
    mats = set()
    for split in ("train", "val", "test"):
        for size in (16, 32):
            mats |= {s["material"] for s in load(size, split, extra=extra if split == "train" else False)}
    ps = {" ".join(prompt_words(m)) or m for m in mats}
    ps |= {e["prompt"] for e in load_set("E_mat")[0]}
    return sorted(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=2, help="每个提示词渲几张")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/refs")
    ap.add_argument("--extra", default=None,
                    help="训练用的额外训练文件名（如 train_extra_packs_only.json）。"
                         "默认 None = 旧行为；训练带 --extra 时这里必须给同一个文件，否则材质覆盖不全")
    ap.add_argument("--list_only", action="store_true", help="只打印提示词数量后退出（不加载 SDXL）")
    a = ap.parse_args()
    prompts = all_prompts(a.extra or False)
    if a.list_only:
        print(f"提示词共 {len(prompts)}（extra={a.extra}）")
        return
    mine = [(i, p) for i, p in enumerate(prompts) if i % a.nshards == a.shard]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "thumbs").mkdir(exist_ok=True)
    emb_path = a.out / f"emb_shard{a.shard}.pt"
    embs = torch.load(emb_path) if emb_path.exists() else {}
    print(f"提示词共 {len(prompts)}，本分片 {len(mine)}，已完成 {len(embs)}", flush=True)

    from diffusers import StableDiffusionXLPipeline
    from transformers import CLIPModel
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda").eval()
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device="cuda").view(1, 3, 1, 1)

    for n_done, (i, p) in enumerate(mine):
        if p in embs:
            continue
        ims = []
        for kk in range(a.k):
            g = torch.Generator("cuda").manual_seed(a.seed + i + 1000 * kk)
            ims.append(pipe(TMPL.format(p=p), negative_prompt=NEG, num_inference_steps=a.steps,
                            generator=g, height=1024, width=1024).images[0])
        x = torch.stack([torch.from_numpy(np.asarray(im.resize((224, 224), Image.BICUBIC))).permute(2, 0, 1)
                         for im in ims]).float().cuda() / 255.0
        with torch.no_grad():
            e = F.normalize(clip.get_image_features(pixel_values=(x - mean) / std).float(), dim=-1)
        embs[p] = e.cpu()
        ims[0].resize((128, 128), Image.BICUBIC).save(a.out / "thumbs" / f"{p.replace(' ', '_')[:80]}.png")
        if n_done % 20 == 0:
            torch.save(embs, emb_path)
            print(f"[{len(embs)}/{len(mine)}] {p}", flush=True)
    torch.save(embs, emb_path)
    print("done", emb_path, len(embs))


if __name__ == "__main__":
    main()
