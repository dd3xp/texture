"""用 TRD 检查点给评测提示词集出图（与基线同一批提示词、同样每材质 n 张）。

色阶数 k：从训练集 k_used 的经验分布里抽（v1 的做法，写在这里以免事后改）；
颜色条件丢弃（与基线同等条件）。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "eval"))
from trd import TRD, sample                      # noqa: E402
from train_trd import decode, clip_text, TEXT_TMPL, model_from_args   # noqa: E402
from tiles_data import load                      # noqa: E402


def retrieve_batch(mem, text_rows, ks, rng, cb, colours=None, topk=5):
    """检索增强调色板（model/palette_memory.py）：每行取一张真人调色板，返回 (ks, pal_init 码, 精确调色板)。"""
    from train_trd import encode_palette
    from trd import K_MAX
    pals, codes, knew = [], [], []
    for i in range(len(ks)):
        c = None if colours is None else colours[i]
        p, idx = mem.query(text_rows[i], int(ks[i]), rng, colour=c, topk=topk)
        if c is not None:
            p = mem.shifted(idx, c)
        pals.append(p)
        knew.append(len(p))
        row = np.full(K_MAX, len(cb) + 1, np.int64)           # PAD
        row[:len(p)] = encode_palette(p, cb)
        codes.append(row)
    return (torch.tensor(knew, device=ks.device), torch.tensor(np.stack(codes), device=ks.device), pals)


def render(pals, grid):
    g = grid.cpu().numpy()
    return [np.asarray(p, np.uint8)[np.clip(g[j], 0, len(p) - 1)] for j, p in enumerate(pals)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--set", default="E_mat")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=32, help="采样批次（CFG 要两次前向，共享卡上别开太大）")
    ap.add_argument("--pal_top_p", type=float, default=0.9)
    ap.add_argument("--choice_temp", type=float, default=4.5)
    ap.add_argument("--refine", type=int, default=0, help="解码后块 Gibbs 精修轮数（trd.sample）")
    ap.add_argument("--refine_frac", type=float, default=0.25)
    ap.add_argument("--refine_temp", type=float, default=0.7)
    ap.add_argument("--tag", default=None, help="输出目录名（默认由参数拼出）")
    ap.add_argument("--refs", type=Path, default=ROOT / "experiments/refs",
                    help="v3 模型的参考图嵌入目录（render_refs.py 输出）；推理用第 0 张渲染")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--pal_mode", choices=["model", "retrieve", "retrieve_model"], default="model",
                    help="retrieve = 检索增强调色板（按文本检索真人调色板，TRD 只生成结构）；"
                         "retrieve_model = TRD 先出一张定颜色，再按 文本+该颜色 检索真人调色板、平移到该颜色、重生成结构")
    ap.add_argument("--ret_topk", type=int, default=5)
    ap.add_argument("--colour_task", action="store_true",
                    help="按 eval/colour_task.py 的目标（每张真人参照瓦片的材质名 + 平均色）出图，颜色条件打开")
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed)
    ck = torch.load(a.run / a.ckpt, map_location=dev)
    args = ck["args"]
    cb = np.load(a.run / "codebook.npy")
    model = model_from_args(args, drop=0.0).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    from prompts import load_set
    prompts, _ = load_set(a.set)
    temb = clip_text([TEXT_TMPL.format(p=e["prompt"]) for e in prompts], dev).to(dev)
    rembs = None
    if model.ref_proj is not None:
        ref = {}
        for f in sorted(a.refs.glob("emb_shard*.pt")):
            ref.update(torch.load(f))
        miss = [e["prompt"] for e in prompts if e["prompt"] not in ref]
        if miss:
            raise SystemExit(f"参考图嵌入缺 {len(miss)} 个提示词（先跑 render_refs.py）：{miss[:5]}")
        rembs = torch.stack([ref[e["prompt"]][0] for e in prompts]).to(dev)
    kdist = np.bincount([s["k_used"] for s in load(16, "train")], minlength=17).astype(float)
    kdist /= kdist.sum()
    rng = np.random.default_rng(a.seed)
    mem = None
    if a.pal_mode != "model":
        from palette_memory import PaletteMemory
        mem = PaletteMemory(dev)

    tag = a.tag or f"TRD_{a.run.name}_cfg{a.cfg}_t{a.temp}_p{a.pal_top_p}_{a.set}"
    if a.colour_task:                               # 任务本身：区域颜色 + 材质名（eval/colour_task.py）
        from colour_task import targets
        from PIL import Image
        T = targets(a.set, a.size if a.size in (16, 32) else 16)
        pidx = {e["prompt"]: i for i, e in enumerate(prompts)}
        out = ROOT / "experiments/colour" / tag / str(a.size)
        out.mkdir(parents=True, exist_ok=True)
        ks = torch.tensor(rng.choice(17, len(T), p=kdist), device=dev)
        ti = torch.tensor([pidx[t["prompt"]] for t in T], device=dev)
        col = torch.tensor([[*(np.array(t["rgb"]) / 255.0), 1.0] for t in T], dtype=torch.float32, device=dev)
        for i in range(0, len(T), a.bs):
            sl = slice(i, i + a.bs)
            kb, pinit, pals = ks[sl], None, None
            if mem is not None:
                kb, pinit, pals = retrieve_batch(mem, temb[ti[sl]], kb, rng, cb,
                                                 colours=[t["rgb"] for t in T[sl]], topk=a.ret_topk)
            pal, grid = sample(model, temb[ti[sl]], kb, n=a.size, color=col[sl], steps=a.steps,
                               cfg=a.cfg, temp=a.temp, pal_top_p=a.pal_top_p, choice_temp=a.choice_temp,
                               refine=a.refine, refine_frac=a.refine_frac, refine_temp=a.refine_temp,
                               ref=None if rembs is None else rembs[ti[sl]], pal_init=pinit)
            imgs = decode(pal, grid, cb) if pals is None else render(pals, grid)
            for j, t in enumerate(T[sl]):
                Image.fromarray(imgs[j]).save(out / f"{t['slug']}_{t['j']}.png")
        print("->", out, len(T))
        return
    out = (a.out or ROOT / "experiments/baselines") / tag / str(a.size)
    out.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    for kk in range(a.n):
        ks = torch.tensor(rng.choice(17, len(prompts), p=kdist), device=dev)
        for i in range(0, len(prompts), a.bs):
            sl = slice(i, i + a.bs)
            kb, pinit, pals = ks[sl], None, None
            if mem is not None:
                cols = None
                if a.pal_mode == "retrieve_model":      # 第一遍：TRD 自己定颜色（颜色语义来自模型）
                    p0, g0 = sample(model, temb[sl], kb, n=a.size, steps=a.steps, cfg=a.cfg, temp=a.temp,
                                    pal_top_p=a.pal_top_p, choice_temp=a.choice_temp)
                    cols = [im.reshape(-1, 3).mean(0) for im in decode(p0, g0, cb)]
                kb, pinit, pals = retrieve_batch(mem, temb[sl], kb, rng, cb, colours=cols, topk=a.ret_topk)
            pal, grid = sample(model, temb[sl], kb, n=a.size, steps=a.steps,
                               cfg=a.cfg, temp=a.temp, pal_top_p=a.pal_top_p,
                               choice_temp=a.choice_temp, refine=a.refine, refine_frac=a.refine_frac,
                               refine_temp=a.refine_temp, ref=None if rembs is None else rembs[sl],
                               pal_init=pinit)
            imgs = decode(pal, grid, cb) if pals is None else render(pals, grid)
            for j, e in enumerate(prompts[sl]):
                slug = e["material"].rsplit(".", 1)[0]
                Image.fromarray(imgs[j]).save(out / f"{slug}_{kk}.png")
        print(f"sample {kk + 1}/{a.n} done", flush=True)
    print("->", out.parent)


if __name__ == "__main__":
    main()
