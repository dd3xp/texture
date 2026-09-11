"""训练 Token-Critic（Lezama et al., ECCV 2022）：给 TRD 的采样当"改错器"。

判官在验证集上：画师对 B2 胜 66%，TRD 单张 27–35%；分布指标已追平甚至超过画师附近，差的是**单张的完成度**。
MaskGIT 的格子一旦定下就不能改，早期定错的孤立像素、断掉的砖缝会一直留着。Critic 学"哪些格子像是被采样替换的"，
采样时（`trd.sample_critic`）每一步把最可疑的格子重新掩上再采，包括之前定下的。

训练数据（生成器冻结）：真人网格按 MaskGIT 余弦日程随机掩一部分 → 冻结的 TRD 一步采满被掩格 → 标签 = 该格是否被替换过。
条件与生成器一致（文本、真人调色板、色数、结构范例、来源）；critic 自己比生成器小（默认 d=256、8 层）。

    python model/train_critic.py --gen_run runs/trd_v7 --out runs/critic_v7
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "eval"))
from trd import TRD, K_MAX, mask_ratio                                         # noqa: E402
from train_trd import to_tensors, augment, clip_text, text_prompt, model_from_args   # noqa: E402
from tiles_data import load                                                   # noqa: E402


def load_critic(path, dev):
    """gen_trd.py --critic 用：按训练时的参数重建 critic。"""
    ck = torch.load(path, map_location=dev)
    a, g = ck["args"], ck["gen_args"]
    m = TRD(int(g.get("codes", 512)), d=int(a["d"]), depth=int(a["depth"]), heads=int(a["heads"]), drop=0.0,
            bias_freqs=int(g.get("bias_freqs", 8)), level_emb=bool(g.get("level_emb", True)),
            bias_hidden=int(g.get("bias_hidden", 128)), n_exemplars=int(g.get("n_ex", 0) or 0),
            n_domains=2 if g.get("domain") else 0, critic=True).to(dev)
    m.load_state_dict(ck["model"])
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_run", type=Path, required=True)
    ap.add_argument("--gen_ckpt", default="last.pt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--batch32", type=int, default=32)
    ap.add_argument("--p32", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=1000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(0)
    torch.set_num_threads(4)

    ck = torch.load(a.gen_run / a.gen_ckpt, map_location=dev)
    gcfg = ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"])
    gen = model_from_args(gcfg, drop=0.0).to(dev)
    gen.load_state_dict(ck["model"])
    gen.eval().requires_grad_(False)
    cb = np.load(a.gen_run / "codebook.npy")
    codes = int(gcfg.get("codes", 512))
    ext = gcfg.get("extra_file", "train_extra.json") if gcfg.get("extra") else False

    train, val = load(16, "train", extra=ext), load(16, "val")
    train32 = load(32, "train", extra=ext)
    mats = sorted({s["material"] for s in train + val + train32})
    temb = clip_text([text_prompt(m) for m in mats], dev)
    tindex = {m: i for i, m in enumerate(mats)}
    T, V = to_tensors(train, cb, codes, tindex, temb), to_tensors(val, cb, codes, tindex, temb)
    T32 = to_tensors(train32, cb, codes, tindex, temb)
    BANK = None
    if gen.n_ex:
        from exemplars import ExemplarBank
        BANK = ExemplarBank(train, {m: temb[tindex[m]] for m in mats}, dev)
        for Dd, smp in ((T, train), (V, val), (T32, train32)):
            Dd["ex_cand"] = BANK.candidates(smp)
    for Dd in (T, V, T32):
        for kk in Dd:
            Dd[kk] = Dd[kk].to(dev)

    critic = TRD(codes, d=a.d, depth=a.depth, heads=a.heads, drop=0.1,
                 bias_freqs=int(gcfg.get("bias_freqs", 8)), level_emb=bool(gcfg.get("level_emb", True)),
                 bias_hidden=int(gcfg.get("bias_hidden", 128)), n_exemplars=gen.n_ex,
                 n_domains=2 if gcfg.get("domain") else 0, critic=True).to(dev)
    opt = torch.optim.AdamW(critic.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))
    cfg = {kk: (str(v) if isinstance(v, Path) else v) for kk, v in vars(a).items()}
    json.dump(cfg, open(a.out / "config.json", "w"), indent=1)
    print(f"critic 参数 {sum(p.numel() for p in critic.parameters()) / 1e6:.1f}M；train {len(train)}+{len(train32)}", flush=True)

    def batch(D, idx, train_mode):
        idx = idx.to(dev)
        g = D["grid"][idx]
        if train_mode:
            g = augment(g)
        B, N = g.shape[0], g.shape[1]
        pal, k, t = D["pal"][idx], D["k"][idx], D["text"][idx]
        c = gen.null_color[None].expand(B, -1)
        ex = BANK.draw(D["ex_cand"][idx], gen.n_ex, train_mode, 0.3) if BANK is not None else None
        dom = D["dom"][idx] if gcfg.get("domain") else None
        r = mask_ratio(torch.rand(B, device=dev))                       # MaskGIT 余弦日程
        m = torch.rand(B, N, N, device=dev) < r[:, None, None]
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
            _, lg = gen(pal, g.masked_fill(m, gen.GRID_MASK), k, t, c, None, None, ex, dom)
        ok = torch.arange(K_MAX, device=dev)[None] < k[:, None]
        lg = lg.float().masked_fill(~ok[:, None, None, :], float("-inf"))
        sg = torch.multinomial(F.softmax(lg, -1).view(-1, K_MAX), 1).view(B, N, N)
        corrupt = torch.where(m, sg, g)
        return pal, corrupt, k, t, c, ex, dom, m.float()

    def step_loss(D, idx, train_mode):
        pal, corrupt, k, t, c, ex, dom, y = batch(D, idx, train_mode)
        with torch.autocast(dev, dtype=torch.bfloat16):
            _, logit = critic(pal, corrupt, k, t, c, None, None, ex, dom)
        loss = F.binary_cross_entropy_with_logits(logit.float(), y)
        acc = ((logit.float() > 0).float() == y).float().mean()
        return loss, acc

    t0, log = time.time(), []
    for step in range(a.steps + 1):
        critic.train()
        use32 = torch.rand(()).item() < a.p32
        D, bs = (T32, a.batch32) if use32 else (T, a.batch)
        loss, acc = step_loss(D, torch.randint(0, len(D["k"]), (bs,)), True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % a.eval_every == 0:
            critic.eval()
            with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(123)
                vl, va = step_loss(V, torch.arange(min(256, len(V["k"]))), False)
            rec = {"step": step, "train": loss.item(), "train_acc": acc.item(), "val": vl.item(), "val_acc": va.item(),
                   "min": (time.time() - t0) / 60}
            log.append(rec)
            print(json.dumps(rec), flush=True)
            json.dump(log, open(a.out / "log.json", "w"), indent=0)
            torch.save({"model": critic.state_dict(), "args": cfg, "gen_args": gcfg, "step": step}, a.out / "last.pt")


if __name__ == "__main__":
    main()
