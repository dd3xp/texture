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


N_NAME = 30
FREE_K = True      # 检索调色板时不限色数（见 palette_memory.PaletteMemory.query）；--ret_k sample 改回旧做法


def retrieve_batch(mem, text_rows, ks, rng, cb, colours=None, topk=5, t16_rows=None, xpal_temp=None):
    """检索增强调色板（model/palette_memory.py）：每行取一张真人调色板，返回 (ks, pal_init 码, 精确调色板)。"""
    from train_trd import encode_palette
    from trd import K_MAX
    pals, codes, knew = [], [], []
    for i in range(len(ks)):
        c = None if colours is None else colours[i]
        p, idx = mem.query(text_rows[i], None if FREE_K else int(ks[i]), rng, colour=c, topk=topk,
                           text16=None if t16_rows is None else t16_rows[i], n_name=N_NAME,
                           xpal_temp=xpal_temp)
        if c is not None:
            p = mem.shifted(idx, c)
        pals.append(p)
        knew.append(len(p))
        row = np.full(K_MAX, len(cb) + 1, np.int64)           # PAD
        row[:len(p)] = encode_palette(p, cb)
        codes.append(row)
    return (torch.tensor(knew, device=ks.device), torch.tensor(np.stack(codes), device=ks.device), pals)


def xquery(prompts, img_dir, dev):
    """跨模态检索的查询向量：材质名的 CLIP-B/16 文本嵌入；给了 img_dir 再加上该材质大模型渲染图的 B/16 图像嵌入。"""
    import torch.nn.functional as F
    from palette_memory import clip16, clip16_images, clip16_texts
    m, tok = clip16(dev)
    q = clip16_texts([e["prompt"] for e in prompts], dev, m, tok)
    if img_dir is not None:
        from PIL import Image
        ims = []
        for e in prompts:
            slug = e["material"].rsplit(".", 1)[0]
            f = next((img_dir / n for n in (f"{slug}_0.png", f"{slug}.png") if (img_dir / n).exists()), None)
            ims.append(np.asarray(Image.open(f).convert("RGB")) if f else None)
        have = [i for i, t in enumerate(ims) if t is not None]
        emb = clip16_images([ims[i] for i in have], dev, m)
        q[have] = F.normalize(q[have] + emb, dim=-1)
    return q


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
    ap.add_argument("--choice_temp", type=float, default=20.0,
                    help="解码顺序的 Gumbel 噪声强度。原默认 4.5（MaskGIT 惯例）在验证集上是错的：按置信度先定死"
                         "最有把握的格子，会把横条纹式周期结构套到所有材质上；20 时 KID 12.8→5.5、CLIP +0.4"
                         "（experiments/eval_v4ct*_Vmat_first.json）")
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
    ap.add_argument("--ret_k", choices=["free", "sample"], default="free",
                    help="free = 色数跟检索到的真人调色板走；sample = 旧做法（先从全局分布抽 k 再限定 k 色）")
    ap.add_argument("--no_ex", action="store_true", help="v7 模型不给结构范例（消融）")
    ap.add_argument("--ex_cfg", type=float, default=None, help="结构范例单独的引导强度（trd.sample）")
    ap.add_argument("--critic", type=Path, default=None,
                    help="Token-Critic 检查点（model/train_critic.py）；给了就用 trd.sample_critic（需检索调色板）")
    ap.add_argument("--critic_noise", type=float, default=1.0)
    ap.add_argument("--cascade", type=float, default=None,
                    help="24/32px 由粗到细：先按同样条件生成 16px，最近邻放大到目标尺寸，保留这个比例的格子，其余由 TRD 在目标尺寸上补")
    ap.add_argument("--dom_w", type=float, default=0.0, help="来源引导强度（v9：往 SDXL 字面描绘那边推）")
    ap.add_argument("--ret_nname", type=int, default=30, help="--xmodal：先按名字取这么多条，再按图文相似度挑")
    ap.add_argument("--ex_keep", type=int, default=8, help="--xmodal 时结构范例只在最典型的这么多张里抽")
    ap.add_argument("--xpal_temp", type=float, default=None,
                    help="--xmodal 调色板改用软典型度：在 ret_nname 个候选上按 softmax(图文相似度/温度) 抽，"
                         "而不是硬取前 ret_topk。不给 = 原来的硬取")
    ap.add_argument("--ex_from_dir", type=Path, default=None,
                    help="结构范例的第 0 个槽换成这个目录里该材质第 0 张瓦片（如 B1 的 16px：SDXL 的布局当一个范例）")
    ap.add_argument("--xquery_img", type=Path, default=None,
                    help="--xmodal 的查询再加上大模型渲染的图像嵌入：给一个方法目录（如 experiments/baselines_val/B1/32），"
                         "取每材质第 0 张的 CLIP-B/16 图像嵌入与文本嵌入相加（SDXL 引导的检索；推理多一次渲染，与 B1/B2 同开销）")
    ap.add_argument("--xmodal_what", choices=["both", "ex", "pal"], default="both",
                    help="--xmodal 作用在哪：both = 调色板与结构范例都按图文相似度挑；ex = 只范例；pal = 只调色板")
    ap.add_argument("--xmodal", action="store_true",
                    help="跨模态检索：调色板与结构范例按'真人瓦片 ↔ 材质名'的 CLIP-B/16 图文相似度挑最典型的（评测用 B/32）")
    ap.add_argument("--align", type=float, default=None,
                    help="对齐分数条件的分位数（模型用 --align_clip 训练时才有效），如 0.9")
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
    XPAL = a.xmodal and a.xmodal_what in ("both", "pal")
    global FREE_K, N_NAME
    FREE_K = a.ret_k == "free"
    N_NAME = a.ret_nname

    def AL(n):
        if a.align is None or model.align_proj is None:
            return None
        return torch.tensor([[a.align, 1.0]], device=dev).expand(n, 2)
    EXC, BANK = None, None
    if getattr(model, "ex_proj", None) is not None and not a.no_ex:     # v7：结构范例（其他画师的同材质瓦片）
        from exemplars import ExemplarBank
        from train_trd import text_prompt
        cfg_ = ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"])
        pool = load(16, "train", extra=cfg_.get("extra_file", "train_extra.json"))   # 与该 run 训练时同一个范例库
        pm = sorted({s["material"] for s in pool})
        pe = clip_text([text_prompt(m) for m in pm], dev)
        XEX = a.xmodal and a.xmodal_what in ("both", "ex")
        BANK = ExemplarBank(pool, {m: pe[i] for i, m in enumerate(pm)}, dev, C=64 if XEX else 16)
        EXC = BANK.candidates([{"material": e["material"]} for e in prompts], emb=temb)
        if XEX:
            from palette_memory import clip16_images, clip16_texts
            EXC = BANK.rank_xmodal(EXC, clip16_images([s["palette"][s["idx"]] for s in pool], dev),
                                   xquery(prompts, a.xquery_img, dev), keep=a.ex_keep)

    EXT = None
    if EXC is not None and a.ex_from_dir is not None:        # 大模型瓦片 → 归一化色阶网格（与 exemplars.levels 同一换算）
        from PIL import Image
        from tiles_data import canonicalise
        from exemplars import levels
        rows_ = []
        for e in prompts:
            slug = e["material"].rsplit(".", 1)[0]
            f = next((a.ex_from_dir / n for n in (f"{slug}_0.png", f"{slug}.png") if (a.ex_from_dir / n).exists()), None)
            if f is None:
                rows_.append(np.full((16, 16), -1, np.int64))
                continue
            t = np.asarray(Image.open(f).convert("RGB").resize((16, 16), Image.NEAREST))
            cols, inv = np.unique(t.reshape(-1, 3), axis=0, return_inverse=True)
            g, p = canonicalise(inv.reshape(16, 16).astype(np.int64), cols.astype(np.uint8))
            rows_.append(levels({"idx": g, "k_used": len(p)}))
        EXT = torch.tensor(np.stack(rows_), device=dev)

    def EX(rows):
        if EXC is None:
            return None
        ex = BANK.draw(EXC[rows], model.n_ex, False)
        if EXT is not None:
            ex[:, 0] = EXT[rows]
        return ex
    CRITIC = None
    if a.critic is not None:
        from train_critic import load_critic
        CRITIC = load_critic(a.critic, dev)
    mem = None
    if a.pal_mode != "model":
        from palette_memory import PaletteMemory
        mem = PaletteMemory(dev)
        if a.xmodal:
            from palette_memory import clip16_texts
            mem.enable_xmodal(dev)
            T16 = xquery(prompts, a.xquery_img, dev)

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
                                                 colours=[t["rgb"] for t in T[sl]], topk=a.ret_topk,
                                                 t16_rows=T16[ti[sl]] if XPAL else None,
                                                 xpal_temp=a.xpal_temp)
            if CRITIC is not None and pinit is not None:
                from trd import sample_critic
                pal, grid = sample_critic(model, CRITIC, temb[ti[sl]], kb, pinit, n=a.size, color=col[sl],
                                          steps=a.steps, temp=a.temp, cfg=a.cfg, noise=a.critic_noise, ex=EX(ti[sl]))
                imgs = render(pals, grid)
                for j, t in enumerate(T[sl]):
                    Image.fromarray(imgs[j]).save(out / f"{t['slug']}_{t['j']}.png")
                continue
            pal, grid = sample(model, temb[ti[sl]], kb, n=a.size, color=col[sl], steps=a.steps,
                               cfg=a.cfg, temp=a.temp, pal_top_p=a.pal_top_p, choice_temp=a.choice_temp,
                               refine=a.refine, refine_frac=a.refine_frac, refine_temp=a.refine_temp,
                               ref=None if rembs is None else rembs[ti[sl]], pal_init=pinit, align=AL(len(kb)),
                               ex=EX(ti[sl]), ex_cfg=a.ex_cfg, dom_w=a.dom_w)
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
                                    pal_top_p=a.pal_top_p, choice_temp=a.choice_temp, align=AL(len(kb)),
                                    ex=EX(torch.arange(len(prompts))[sl].to(dev)))
                    cols = [im.reshape(-1, 3).mean(0) for im in decode(p0, g0, cb)]
                kb, pinit, pals = retrieve_batch(mem, temb[sl], kb, rng, cb, colours=cols, topk=a.ret_topk,
                                                 t16_rows=T16[sl] if XPAL else None, xpal_temp=a.xpal_temp)
            gi = None
            if a.cascade is not None and a.size != 16:          # 由粗到细：16px 结构 → 放大 → 部分保留
                exs = EX(torch.arange(len(prompts))[sl].to(dev))
                pal16, g16 = sample(model, temb[sl], kb, n=16, steps=a.steps, cfg=a.cfg, temp=a.temp,
                                    pal_top_p=a.pal_top_p, choice_temp=a.choice_temp, pal_init=pinit,
                                    align=AL(len(kb)), ex=exs, ex_cfg=a.ex_cfg, dom_w=a.dom_w)
                if pinit is None:
                    pinit = pal16
                ix = (torch.arange(a.size, device=dev) * 16) // a.size
                if getattr(model, "coarse_emb", None) is not None:  # v10：16px 结果当"粗网格"条件，目标尺寸从头生成
                    lvl = torch.round(15.0 * g16.float() / (kb[:, None, None].float() - 1).clamp(min=1)).long().clamp(0, 15)
                    CZ = lvl[:, ix][:, :, ix].contiguous()
                    gi = torch.full_like(CZ, model.GRID_MASK)
                else:
                    CZ = None
                    gi = g16[:, ix][:, :, ix].clone()
                    gi[torch.rand(gi.shape, device=dev) >= a.cascade] = model.GRID_MASK
            if CRITIC is not None and pinit is not None:
                from trd import sample_critic
                pal, grid = sample_critic(model, CRITIC, temb[sl], kb, pinit, n=a.size, steps=a.steps, temp=a.temp,
                                          cfg=a.cfg, noise=a.critic_noise, ex=EX(torch.arange(len(prompts))[sl].to(dev)))
            elif gi is not None:
                pal, grid = sample(model, temb[sl], kb, n=a.size, steps=a.steps, cfg=a.cfg, temp=a.temp,
                                   pal_top_p=a.pal_top_p, choice_temp=a.choice_temp, pal_init=pinit, grid_init=gi,
                                   align=AL(len(kb)), ex=exs, ex_cfg=a.ex_cfg, dom_w=a.dom_w, coarse=CZ)
            else:
                pal, grid = sample(model, temb[sl], kb, n=a.size, steps=a.steps,
                                   cfg=a.cfg, temp=a.temp, pal_top_p=a.pal_top_p,
                                   choice_temp=a.choice_temp, refine=a.refine, refine_frac=a.refine_frac,
                                   refine_temp=a.refine_temp, ref=None if rembs is None else rembs[sl],
                                   pal_init=pinit, align=AL(len(kb)), ex=EX(torch.arange(len(prompts))[sl].to(dev)),
                                   ex_cfg=a.ex_cfg, dom_w=a.dom_w)
            imgs = decode(pal, grid, cb) if pals is None else render(pals, grid)
            for j, e in enumerate(prompts[sl]):
                slug = e["material"].rsplit(".", 1)[0]
                Image.fromarray(imgs[j]).save(out / f"{slug}_{kk}.png")
        print(f"sample {kk + 1}/{a.n} done", flush=True)
    print("->", out.parent)


if __name__ == "__main__":
    main()
