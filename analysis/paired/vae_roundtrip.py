"""训练目标本身是不是坏的？——VAE 对「放大的 16px 瓦片」能不能重建。

方法二第一次训练塌了（`docs/loop.md`）。标准解释是学习率过高，正在用 lr 1e-5 重试。
但还有一个可能，若成立则**降学习率注定无用**：

  训练目标是把 16×16 真人瓦片用 NEAREST 放大到 1024，于是每个「像素」是 64×64 的
  硬边色块。这种图对 SDXL 的 VAE 是**严重离分布**的输入。扩散训练学的是在
  **潜空间**里去噪，若 VAE 编码-解码这类图就已经失真，那么 LoRA 被要求拟合的
  是一组本身就坏掉的潜变量——学习率调到多低都救不回来。

**测法**：把真人瓦片放大 -> VAE 编码 -> 解码 -> 与原图比。
对照组用 SDXL 自己生成的图（分布内），看两者的重建误差差多少。

--- 判据（跑之前写下）---
若瓦片的重建误差**显著高于**生成图（中位差 ≥2 倍，且逐样本符号检验 p<0.05），
则训练目标确实离分布，**这套「放大 16px 瓦片当目标」的配置有结构性缺陷**，
第二次训练即便不塌也难指望——该结论要写进停止规则的依据里。
若两者相当，则塌陷更可能纯是优化问题，降学习率是对症的。

纯诊断，不产生任何论文主张。
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tools", "analysis", "analysis/metric"):
    sys.path.insert(0, str(ROOT / sub))
from exact import binom_test                                   # noqa: E402

BASE = "stabilityai/stable-diffusion-xl-base-1.0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="每组样本数")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--gen-dir", type=Path, default=ROOT / "experiments/figqual",
                    help="分布内对照：已有的 SDXL 渲染图")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/vae_roundtrip.json")
    a = ap.parse_args()

    import torch
    from diffusers import AutoencoderKL
    from PIL import Image

    rng = np.random.default_rng(a.seed)
    ds = json.loads((ROOT / "data/tiles/dataset_k16.json").read_text())
    tiles = [s for s in ds["samples"] if s["size"] == 16]
    pick = rng.choice(len(tiles), size=min(a.n, len(tiles)), replace=False)

    gen_files = sorted(a.gen_dir.rglob("hires_*.png"))
    if not gen_files:
        raise SystemExit(f"找不到分布内对照图（{a.gen_dir}/*/hires_*.png）")
    print(f"瓦片 {len(pick)} 张；分布内对照 {len(gen_files)} 张", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float32
    vae = AutoencoderKL.from_pretrained(BASE, subfolder="vae", torch_dtype=dt).to(dev).eval()

    @torch.no_grad()
    def rt_err(img: "Image.Image") -> float:
        """重建误差：解码结果与输入的逐像素 L1（0-1 尺度）。"""
        x = torch.from_numpy(np.asarray(img.convert("RGB"))).float().permute(2, 0, 1)[None]
        x = (x / 127.5 - 1.0).to(dev, dtype=dt)
        lat = vae.encode(x).latent_dist.mean
        y = vae.decode(lat).sample
        return float((x - y).abs().mean().item() / 2.0)

    tile_err, gen_err = [], []
    for i, k in enumerate(pick):
        s = tiles[int(k)]
        pal = np.array(s["palette"], np.uint8)
        idx = np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16)
        img = Image.fromarray(pal[idx]).resize((a.res,) * 2, Image.NEAREST)
        tile_err.append(rt_err(img))
        if (i + 1) % 8 == 0:
            print(f"  瓦片 {i+1}/{len(pick)}", flush=True)
    for i, f in enumerate(gen_files[:a.n]):
        img = Image.open(f).convert("RGB").resize((a.res,) * 2, Image.LANCZOS)
        gen_err.append(rt_err(img))

    mt, mg = statistics.median(tile_err), statistics.median(gen_err)
    a.out.write_text(json.dumps({"tile_err": tile_err, "gen_err": gen_err,
                                 "median_tile": mt, "median_gen": mg},
                                ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n重建误差中位：放大的真人瓦片 {mt:.4f}   SDXL 生成图 {mg:.4f}"
          f"   -> 比值 {mt/mg:.2f}×")
    # 逐样本比较：两组独立，配对不了，用「瓦片误差是否高于生成图中位」的符号检验
    above = sum(1 for e in tile_err if e > mg)
    ps = binom_test(above, len(tile_err))
    print(f"高于生成图中位的瓦片：{above}/{len(tile_err)}，符号检验 p={ps:.3g}")
    if mt / mg >= 2.0 and ps < 0.05:
        print("\n判读：**训练目标确实严重离分布** —— VAE 重建这类图的误差是生成图的"
              f"{mt/mg:.1f} 倍。")
        print("      「放大 16px 瓦片当目标」这套配置有结构性缺陷，降学习率救不回来；")
        print("      第三次若要做，必须换数据形式（例如用 64px 真人瓦片、或在更低分辨率训练）。")
    else:
        print("\n判读：重建误差与分布内相当，塌陷更可能是纯优化问题，降学习率是对症的。")


if __name__ == "__main__":
    main()
