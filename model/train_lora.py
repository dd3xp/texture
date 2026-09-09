"""方法二：在真人瓦片上微调 SDXL，把「结构单元惯例」教给生成器本身。

**为什么这条没被 A4 排除。** A4 训的是掩码预测模型，损失是**逐格 cross-entropy**
（`model/train.py:52`）。而本项目自己的核心发现（B1/B7）是：这个任务
**不存在逐像素目标**——独立画师逐格一致率 0.098，配对零假设 0.086。
拿一个假设「存在唯一正确像素」的目标去训一个没有唯一答案的任务，注定失败，
论文 §3.2 讲的就是这件事。所以被排除的是**逐像素监督**，不是「学习」。
扩散去噪损失是分布式目标，不作这个假设——从没被试过。

**要教的是什么。** §5.2 量到：生成器每图画约 25 个结构单元，真人约 3.2。
裁剪是事后把这个失配补回来。这里换个做法：把真人瓦片放大成 1024 当训练目标，
让模型直接学会「一张图里只画几个大单元」。若学成，裁剪就不再必要。

**数据许可干净**：ContentDB 上 105 个 CC0/CC-BY/MIT/Apache 等许可的材质包，
Minecraft 系（Faithful 等）已排除（`docs/dataset.md`）。
按**材质包**留出（数据集自带 val/test 划分），评测提示词不来自训练过的包。

--- 判据（跑之前写下并 commit）---
操作检验：微调后模型在 1024 上的**单元数中位**必须落到基线的一半以下
        （基线 27.7，故阈值 13.9），且逐提示词配对符号检验 p<0.05。
        不过 = 没学到惯例，主判据不评估，如实记为训练失败。
主判据：**留出提示词**上，经验证判官（claude-opus-5，正反去偏、不一致弃用）
        偏好「微调直出、不裁」胜过「基线 + 裁剪」>50% 且二项 p<0.05
        -> 学到的惯例强过事后裁剪，这是一个**方法**贡献。
次判据（描述性）：「微调 + 裁剪」对「基线 + 裁剪」，看两者是否叠加。
证伪：主判据不成立 -> 学惯例不比裁剪好，如实写，不改判据重跑。
方向性限定：判官压缩效应，为正作下界；为负只能说「没测到大效应」。

跑法（GPU 机器）：
  python model/train_lora.py --steps 1500 --out runs/lora_convention
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis/metric"))

TMPL = ("pixel art, {p}, top-down seamless tileable game texture, "
        "flat lighting, no shadows, orthographic, chunky large pixels")
BASE = "stabilityai/stable-diffusion-xl-base-1.0"


def load_tiles(size_filter=(16,)):
    """真人瓦片 -> (RGB 数组, 材质名)。只用 16px：惯例最极端、也是目标分辨率。"""
    from materials import prompt_for
    ds = json.loads((ROOT / "data/tiles/dataset_k16.json").read_text())
    train, held = [], []
    for s in ds["samples"]:
        if s["size"] not in size_filter:
            continue
        n = s["size"]
        pal = np.array(s["palette"], np.uint8)
        idx = np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(n, n)
        rec = (pal[idx], prompt_for(s["material"]), s["material"], s["pack"])
        # **用样本自带的 `split` 字段**，不要去解析 material。
        # 第一版按 "<pack>__<file>" 拆 material 名，但 material 只是文件名、
        # 不含包前缀，于是每一张都落进训练集 —— 留出等于没做，评测会泄漏。
        # 样本里本来就有 `pack` 与 `split`（split ∈ train/val/test，按包切）。
        (train if s.get("split", "train") == "train" else held).append(rec)
    return train, held


def upscale(rgb: np.ndarray, res: int):
    """NEAREST 放大到训练分辨率——必须保住硬边，插值会把「像素画」这一点抹掉。"""
    from PIL import Image
    return Image.fromarray(rgb.astype(np.uint8)).resize((res, res), Image.NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/lora_convention")
    a = ap.parse_args()

    import torch
    from diffusers import StableDiffusionXLPipeline, DDPMScheduler
    from peft import LoraConfig

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    train, held = load_tiles()
    print(f"训练瓦片 {len(train)}，留出 {len(held)}"
          f"（按材质包切分，留出集不参与训练）", flush=True)
    if not train:
        raise SystemExit("没有训练数据")

    # 冻结的基模走 bf16、只有 LoRA 参数保持 fp32：这是 diffusers 的标准配方，
    # fp32 装整个 SDXL 光权重就 10G，而这台卡是**共享的**，别人占了大半。
    dt = torch.bfloat16
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE, torch_dtype=dt, variant="fp16", use_safetensors=True)
    unet, vae = pipe.unet, pipe.vae
    tok1, tok2 = pipe.tokenizer, pipe.tokenizer_2
    te1, te2 = pipe.text_encoder, pipe.text_encoder_2
    sched = DDPMScheduler.from_pretrained(BASE, subfolder="scheduler")

    dev = "cuda"
    vae.to(dev, dtype=dt).requires_grad_(False).eval()
    te1.to(dev, dtype=dt).requires_grad_(False).eval()
    te2.to(dev, dtype=dt).requires_grad_(False).eval()
    unet.to(dev, dtype=dt)
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(
        r=a.rank, lora_alpha=a.rank, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    unet.enable_gradient_checkpointing()
    # LoRA 参数升回 fp32：bf16 的 AdamW 更新在小 lr 下会被舍入吃掉
    for prm in unet.parameters():
        if prm.requires_grad:
            prm.data = prm.data.float()
    params = [p for p in unet.parameters() if p.requires_grad]
    print(f"LoRA 可训练参数 {sum(p.numel() for p in params)/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr)

    @torch.no_grad()
    def encode_prompt(text):
        out = []
        for tok, te in ((tok1, te1), (tok2, te2)):
            ids = tok(text, padding="max_length", max_length=tok.model_max_length,
                      truncation=True, return_tensors="pt").input_ids.to(dev)
            o = te(ids, output_hidden_states=True)
            out.append(o.hidden_states[-2])
            pooled = o[0]
        return torch.cat(out, dim=-1), pooled

    a.out.mkdir(parents=True, exist_ok=True)
    unet.train()
    step, run = 0, 0.0
    while step < a.steps:
        rgb, prompt, _mat, _pack = train[random.randrange(len(train))]
        img = upscale(rgb, a.res)
        x = torch.from_numpy(np.asarray(img)).float().permute(2, 0, 1)[None]
        x = (x / 127.5 - 1.0).to(dev, dtype=dt)
        with torch.no_grad():
            lat = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(lat)
        t = torch.randint(0, sched.config.num_train_timesteps, (1,), device=dev).long()
        noisy = sched.add_noise(lat, noise, t)
        emb, pooled = encode_prompt(TMPL.format(p=prompt))
        add_time = torch.tensor([[a.res, a.res, 0, 0, a.res, a.res]],
                                device=dev, dtype=dt)
        pred = unet(noisy, t, encoder_hidden_states=emb,
                    added_cond_kwargs={"text_embeds": pooled,
                                       "time_ids": add_time}).sample
        loss = torch.nn.functional.mse_loss(pred.float(), noise.float()) / a.accum
        loss.backward()
        run += loss.item() * a.accum
        step += 1
        if step % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
        if step % 100 == 0:
            print(f"  step {step}/{a.steps}  loss {run/100:.4f}", flush=True)
            run = 0.0

    from peft.utils import get_peft_model_state_dict
    sd = get_peft_model_state_dict(unet)
    StableDiffusionXLPipeline.save_lora_weights(str(a.out), sd)
    (a.out / "meta.json").write_text(json.dumps(
        {"steps": a.steps, "rank": a.rank, "lr": a.lr, "res": a.res,
         "n_train": len(train), "n_heldout_tiles": len(held)},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写入 {a.out}", flush=True)


if __name__ == "__main__":
    main()
