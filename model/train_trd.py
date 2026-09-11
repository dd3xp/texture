"""训练 TRD（新架构 v1）。数据：`split=train` 的 16×16 真人瓦片；`split=val` 选检查点。

用法（服务器，离线）：
    HF_HUB_OFFLINE=1 python model/train_trd.py --out runs/trd_v1 --steps 40000
冒烟：
    python model/train_trd.py --smoke        （极小配置跑几十步 + 采样一次）
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
from tiles_data import load, W_LUM                      # noqa: E402
from trd import TRD, K_MAX, training_loss, sample       # noqa: E402
from prompts import prompt_words                        # noqa: E402

TEXT_TMPL = "pixel art texture of {p}"


# ------------------------------------------------------------------ 颜色码本
def kmeans(x, k, iters=25, seed=0):
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)].astype(np.float64)
    for _ in range(iters):
        d = ((x[:, None, :] - c[None]) ** 2).sum(-1) if len(x) * k < 4e7 else None
        if d is None:                                   # 大时分块
            lab = np.concatenate([((x[i:i + 20000, None] - c[None]) ** 2).sum(-1).argmin(1)
                                  for i in range(0, len(x), 20000)])
        else:
            lab = d.argmin(1)
        for j in range(k):
            sel = x[lab == j]
            c[j] = sel.mean(0) if len(sel) else x[rng.integers(len(x))]
    return c


def build_codebook(samples, k):
    cols = np.concatenate([s["palette"].astype(np.float64) for s in samples])
    cb = kmeans(cols, k)
    # 按亮度排序，便于查看
    cb = cb[np.argsort(cb @ W_LUM)]
    err = np.sqrt(((cols[:, None] - cb[None]) ** 2).sum(-1).min(1)).mean()
    return cb.astype(np.float32), float(err)


def encode_palette(pal, cb):
    d = ((pal[:, None].astype(np.float32) - cb[None]) ** 2).sum(-1)
    return d.argmin(1)


# ------------------------------------------------------------------ 文本
def text_prompt(material):
    w = prompt_words(material)
    return TEXT_TMPL.format(p=" ".join(w) if w else material)


@torch.no_grad()
def clip_text(prompts, dev):
    from transformers import CLIPModel, CLIPTokenizer
    name = "openai/clip-vit-base-patch32"
    m = CLIPModel.from_pretrained(name).to(dev).eval()
    tok = CLIPTokenizer.from_pretrained(name)
    out = []
    for i in range(0, len(prompts), 256):
        t = tok(prompts[i:i + 256], padding=True, return_tensors="pt").to(dev)
        out.append(F.normalize(m.get_text_features(**t).float(), dim=-1).cpu())
    del m
    return torch.cat(out)


# ------------------------------------------------------------------ 数据张量
def to_tensors(samples, cb, n_codes, text_index, text_emb):
    B = len(samples)
    pal = np.full((B, K_MAX), n_codes + 1, np.int64)          # PAD
    N = samples[0]["idx"].shape[0]
    grid = np.zeros((B, N, N), np.int64)
    k = np.zeros(B, np.int64)
    col = np.zeros((B, 4), np.float32)
    tix = np.zeros(B, np.int64)
    dom = np.zeros(B, np.int64)
    rgb = np.zeros((B, K_MAX, 3), np.float32)
    hist = np.zeros((B, K_MAX), np.float32)
    for i, s in enumerate(samples):
        kk = s["k_used"]
        pal[i, :kk] = encode_palette(s["palette"], cb)
        grid[i] = s["idx"]
        k[i] = kk
        col[i, :3] = (s["palette"][s["idx"]].reshape(-1, 3).mean(0) / 255.0).astype(np.float32)
        col[i, 3] = 1.0                                   # 标志位：颜色已给出
        rgb[i, :kk] = s["palette"]
        hist[i, :kk] = np.bincount(s["idx"].reshape(-1), minlength=kk)[:kk] / s["idx"].size
        tix[i] = text_index[s["material"]]
        dom[i] = int(str(s.get("pack", "")).endswith("@mod"))
    return {"pal": torch.from_numpy(pal), "grid": torch.from_numpy(grid), "k": torch.from_numpy(k),
            "color": torch.from_numpy(col), "text": text_emb[torch.from_numpy(tix)],
            "rgb": torch.from_numpy(rgb), "hist": torch.from_numpy(hist), "dom": torch.from_numpy(dom)}


def augment(grid):
    """循环平移（每样本独立）+ 水平翻转。纹理可平铺，任意循环平移都是合法样本。"""
    B, N, _ = grid.shape
    dv = grid.device
    dy = torch.randint(0, N, (B,), device=dv)
    dx = torch.randint(0, N, (B,), device=dv)
    ar = torch.arange(N, device=dv)
    rows = (ar[None] - dy[:, None]) % N
    cols = (ar[None] - dx[:, None]) % N
    g = grid[torch.arange(B, device=dv)[:, None, None], rows[:, :, None], cols[:, None, :]]
    flip = torch.rand(B, device=dv) < 0.5
    g[flip] = g[flip].flip(-1)
    return g


def decode(pal_codes, ranks, cb):
    """pal_codes [B,16], ranks [B,N,N] -> RGB uint8 [B,N,N,3]"""
    cbt = torch.as_tensor(cb)
    codes = pal_codes.clamp(max=len(cb) - 1).cpu()
    cols = cbt[codes]                                      # [B,16,3]
    r = ranks.cpu()
    img = torch.gather(cols, 1, r.view(r.shape[0], -1, 1).expand(-1, -1, 3))
    return img.view(*r.shape, 3).round().clamp(0, 255).byte().numpy()


def model_from_args(a, drop=None):
    """从检查点里存的参数建模型（v1 检查点没有 v2 字段，按 v1 默认值补）。"""
    g = (lambda k, d: a.get(k, d)) if isinstance(a, dict) else (lambda k, d: getattr(a, k, d))
    return TRD(int(g("codes", 512)), d=int(g("d", 384)), depth=int(g("depth", 12)),
               heads=int(g("heads", 6)), drop=float(g("drop", 0.1) if drop is None else drop),
               bias_freqs=int(g("bias_freqs", 1)), level_emb=bool(g("level_emb", False)),
               bias_hidden=int(g("bias_hidden", 64)),
               ref_dim=512 if g("refs", None) else 0, align_cond=bool(g("align_clip", "")),
               n_exemplars=int(g("n_ex", 0) or 0), n_domains=2 if g("domain", False) else 0)


@torch.no_grad()
def align_scores(samples, clip_path, dev, bs=256):
    """每张瓦片与其材质名的图文对齐分数（余弦）。用 --align_clip 指定的 CLIP（不是评测用的 B/32），
    提示词与评测同一模板。瓦片最近邻放大到 224。"""
    from transformers import CLIPModel, CLIPTokenizer
    m = CLIPModel.from_pretrained(clip_path).to(dev).eval()
    tok = CLIPTokenizer.from_pretrained(clip_path)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)
    words = [" ".join(prompt_words(s["material"])) or s["material"] for s in samples]
    uniq = sorted(set(words))
    te = []
    for i in range(0, len(uniq), bs):
        t = tok([f"pixel art texture of {w}" for w in uniq[i:i + bs]], padding=True, return_tensors="pt").to(dev)
        te.append(F.normalize(m.get_text_features(**t).float(), dim=-1))
    te = torch.cat(te)
    wi = {w: i for i, w in enumerate(uniq)}
    out = []
    for i in range(0, len(samples), bs):
        x = torch.cat([F.interpolate(torch.from_numpy(s["palette"][s["idx"]]).permute(2, 0, 1)[None].float().to(dev),
                                     size=224, mode="nearest") for s in samples[i:i + bs]]) / 255.0   # 16/32 混批
        e = F.normalize(m.get_image_features(pixel_values=(x - mean) / std).float(), dim=-1)
        out.append((e * te[[wi[w] for w in words[i:i + bs]]]).sum(-1))
    del m
    return torch.cat(out).cpu().numpy()


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "runs/trd_v1")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--codes", type=int, default=512)
    ap.add_argument("--p_text_drop", type=float, default=0.1)
    ap.add_argument("--p_color_drop", type=float, default=0.5)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--sizes", type=int, nargs="+", default=[16],
                    help="训练用的分辨率。32px 真人数据只有 517 张，与 16 混训；选检查点只看 16px val")
    ap.add_argument("--p32", type=float, default=0.3, help="每步取 32px 批次的概率")
    ap.add_argument("--batch32", type=int, default=64, help="32px 序列长 4 倍，批次相应缩小")
    ap.add_argument("--bias_freqs", type=int, default=1, help="v1=1；v2=8（见 trd.ToroidalBias）")
    ap.add_argument("--bias_hidden", type=int, default=64)
    ap.add_argument("--level_emb", action="store_true", help="v2：秩的归一化色阶嵌入")
    ap.add_argument("--pal_aug", type=float, default=0.0,
                    help="调色板颜色抖动幅度：色相 ±pal_aug*60°、亮度 ±pal_aug*50%%（v1=0）")
    ap.add_argument("--pal_smooth", type=float, default=0.0, help="调色板码标签平滑（v1=0）")
    ap.add_argument("--refs", type=Path, default=None,
                    help="v3：参考图嵌入目录（baselines/render_refs.py 的输出，emb_shard*.pt）")
    ap.add_argument("--p_ref_drop", type=float, default=0.3)
    ap.add_argument("--ema", type=float, default=0.0,
                    help="权重指数滑动平均的衰减（0=关；v3 用 0.999）。存盘与选检查点都用 EMA 权重")
    ap.add_argument("--extra", action="store_true",
                    help="并入 data/tiles/train_extra.json（训练包里被'≥4 包'规则丢掉的瓦片；val/test 不变）")
    ap.add_argument("--align_clip", default="",
                    help="对齐分数条件：用这个 CLIP（本地目录，如 weights/clip-vit-base-patch16）给训练瓦片打图文对齐分")
    ap.add_argument("--p_align_drop", type=float, default=0.5)
    ap.add_argument("--n_ex", type=int, default=0,
                    help="v7 结构范例数（model/exemplars.py：同材质、其他画师的 16px 真人瓦片）")
    ap.add_argument("--p_ex_drop", type=float, default=0.3)
    ap.add_argument("--extra_file", default="train_extra.json",
                    help="--extra 用哪个文件（train_extra_packs_only.json = 不含模组）")
    ap.add_argument("--domain", action="store_true",
                    help="来源条件（材质包 0 / 模组 1）；推理默认 0（材质包画风）")
    ap.add_argument("--save_at", type=int, nargs="*", default=[],
                    help="在这些步额外存 step_<N>.pt（选检查点看采样指标，不看验证损失）")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.batch, a.d, a.depth, a.heads, a.codes, a.eval_every = 40, 16, 64, 2, 2, 64, 20
        a.out = ROOT / "runs/trd_smoke"
    a.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    torch.set_num_threads(4)                              # 共享机器：别吃满所有 CPU 核

    ext = a.extra_file if a.extra else False
    train, val = load(16, "train", extra=ext), load(16, "val")
    test = load(16, "test")
    train32 = load(32, "train", extra=ext) if 32 in a.sizes else []
    val32 = load(32, "val") if 32 in a.sizes else []
    cb, err = build_codebook(train + train32, a.codes)
    np.save(a.out / "codebook.npy", cb)
    print(f"train {len(train)}  val {len(val)}  码本 {a.codes} 色，平均量化误差 {err:.2f}/255", flush=True)

    mats = sorted({s["material"] for s in train + val + test + train32 + val32})
    prompts = [text_prompt(m) for m in mats]
    temb = clip_text(prompts, dev) if not a.smoke else F.normalize(torch.randn(len(mats), 512), dim=-1)
    tindex = {m: i for i, m in enumerate(mats)}
    torch.save({"materials": mats, "prompts": prompts, "emb": temb}, a.out / "text_emb.pt")
    REF, ref_ix = None, None
    if a.refs:
        ref = {}
        for f in sorted(Path(a.refs).glob("emb_shard*.pt")):
            ref.update(torch.load(f))
        keys = sorted(ref)
        kmin = min(v.shape[0] for v in ref.values())
        REF = torch.stack([ref[kk][:kmin] for kk in keys]).to(dev)          # [P, K, 512]
        pindex = {kk: i for i, kk in enumerate(keys)}
        miss = [m for m in mats if (" ".join(prompt_words(m)) or m) not in pindex]
        print(f"参考图嵌入：{len(keys)} 个提示词 × {kmin} 张；缺 {len(miss)} 个材质", flush=True)
        if miss:
            raise SystemExit(f"参考图缺材质（先跑 render_refs.py 补齐）：{miss[:5]}")
        ref_ix = {m: pindex[" ".join(prompt_words(m)) or m] for m in mats}

    T = to_tensors(train, cb, a.codes, tindex, temb)
    V = to_tensors(val, cb, a.codes, tindex, temb)
    T32 = to_tensors(train32, cb, a.codes, tindex, temb) if train32 else None
    V32 = to_tensors(val32, cb, a.codes, tindex, temb) if val32 else None
    BANK = None
    if a.n_ex:                                        # v7 结构范例：候选表按样本预先算好，取样在 GPU 上
        from exemplars import ExemplarBank
        BANK = ExemplarBank(train, {m: temb[tindex[m]] for m in mats}, dev)
        for Dd, smp in ((T, train), (V, val), (T32, train32), (V32, val32)):
            if Dd is not None:
                Dd["ex_cand"] = BANK.candidates(smp)
        nc = (T["ex_cand"] >= 0).sum(1).float()
        print(f"结构范例：库 {len(train)} 张；训练样本候选数 中位 {nc.median().item():.0f}，"
              f"无候选 {(nc == 0).float().mean().item():.1%}", flush=True)
    if a.align_clip:                                  # 对齐分数 → 训练集分位数（val 用训练集的分布换算）
        parts_ = [(T, train), (V, val), (T32, train32), (V32, val32)]
        allsc = align_scores([x for _, smp in parts_ for x in smp], a.align_clip, dev)
        cuts = np.cumsum([0] + [len(smp) for _, smp in parts_])
        sc = [allsc[cuts[i]:cuts[i + 1]] for i in range(len(parts_))]
        ref_sorted = np.sort(np.concatenate([sc[0], sc[2]]))
        np.save(a.out / "align_ref.npy", ref_sorted)
        for (Dd, _), v in zip(parts_, sc):
            if Dd is not None:
                Dd["align"] = torch.tensor(np.searchsorted(ref_sorted, v) / len(ref_sorted), dtype=torch.float32)
        print(f"对齐分数（{a.align_clip}）：train 中位 {np.median(sc[0]):.3f}，val 中位 {np.median(sc[1]):.3f}", flush=True)
    if ref_ix is not None:
        for Dd, smp in ((T, train), (V, val), (T32, train32), (V32, val32)):
            if Dd is not None:
                Dd["ref_ix"] = torch.tensor([ref_ix[x["material"]] for x in smp])
    model = model_from_args(a).to(dev)
    cbt = torch.as_tensor(cb, device=dev)
    print(f"参数 {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))
    cfg = {kk: (str(v) if isinstance(v, Path) else v) for kk, v in vars(a).items()} | {"codebook_err": err}
    json.dump(cfg, open(a.out / "config.json", "w"), indent=1)

    # 数据总共几 MB，整个放 GPU；增广与颜色抖动都在 GPU 上向量化。
    # （首版在 CPU 上逐样本做色相旋转 + 索引增广，训练进程吃掉 34 个 CPU 核——这是共享机器）
    for Dd in (T, V, T32, V32):
        if Dd is not None:
            for kk in Dd:
                Dd[kk] = Dd[kk].to(dev)
    axis = torch.ones(3, device=dev) / math.sqrt(3.0)
    Kx = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], device=dev)
    K2 = Kx @ Kx

    def batch_of(D, idx, train_mode):
        idx = idx.to(dev)
        g = D["grid"][idx]
        if train_mode:
            g = augment(g)
        t = D["text"][idx].clone()
        c = D["color"][idx].clone()
        pal = D["pal"][idx]
        B = len(idx)
        if train_mode and a.pal_aug > 0:
            # 颜色抖动：同一材质各包配色本就不同；逼模型学相对配色而不是背训练包的色值。
            # 色相旋转绕固定灰轴（Rodrigues）：R = I + sinθ·K + (1-cosθ)·K²，逐样本 θ 可一次算完。
            rgb = D["rgb"][idx]
            th = (torch.rand(B, device=dev) * 2 - 1) * a.pal_aug * math.pi / 3
            br = 1 + (torch.rand(B, device=dev) * 2 - 1) * a.pal_aug * 0.5
            s, cth = th.sin()[:, None, None], (1 - th.cos())[:, None, None]
            rgb = (rgb + s * (rgb @ Kx.T) + cth * (rgb @ K2.T)) * br[:, None, None]
            rgb = rgb.clamp(0, 255)
            valid = pal != a.codes + 1
            codes = ((rgb[:, :, None] - cbt[None, None]) ** 2).sum(-1).argmin(-1)
            pal = torch.where(valid, codes, pal)
            mean = (D["hist"][idx][:, :, None] * rgb).sum(1) / 255.0
            c = torch.cat([mean, torch.ones(B, 1, device=dev)], 1)
        if train_mode:
            dt = torch.rand(B, device=dev) < a.p_text_drop
            dc = torch.rand(B, device=dev) < a.p_color_drop
            t[dt] = model.null_text.detach()
            c[dc] = model.null_color.detach()
        else:                                             # 评估：丢颜色条件（与基线同等）
            c[:] = model.null_color.detach()
        al = None
        if "align" in D:
            av = D["align"][idx]
            al = torch.stack([av, torch.ones_like(av)], 1)
            if train_mode:
                al = al * (torch.rand(B, device=dev) >= a.p_align_drop)[:, None]
            else:                                         # 验证损失不给对齐分数，与其他 run 可比
                al = torch.zeros_like(al)
        ex = BANK.draw(D["ex_cand"][idx], a.n_ex, train_mode, a.p_ex_drop) if "ex_cand" in D else None
        dm = D["dom"][idx] if a.domain else None
        if REF is None:
            return [pal, g, D["k"][idx], t, c, None, al, ex, dm]
        pick = torch.randint(0, REF.shape[1], (B,), device=dev) if train_mode             else torch.zeros(B, dtype=torch.long, device=dev)
        r = REF[D["ref_ix"][idx], pick]
        if train_mode:
            r = r * (torch.rand(B, device=dev) >= a.p_ref_drop)[:, None]
        return [pal, g, D["k"][idx], t, c, r, al, ex, dm]

    val_parts = [0.0, 0.0]

    def val_loss(Vd, nval, net=None):
        """固定掩码随机性让各次验证损失可比；fork_rng 隔离，不污染训练的随机序列。"""
        tot, tg, tp, cnt = 0.0, 0.0, 0.0, 0
        devs = [torch.cuda.current_device()] if dev == "cuda" else []
        with torch.random.fork_rng(devices=devs), torch.no_grad(), \
                torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            torch.manual_seed(123)
            for i in range(0, nval, 256):
                idx = torch.arange(i, min(i + 256, nval))
                l, parts = training_loss(net if net is not None else model, *batch_of(Vd, idx, False))
                tot += l.item() * len(idx)
                tg += parts["loss_grid"] * len(idx)
                tp += parts["loss_pal"] * len(idx)
                cnt += len(idx)
        # 分开报结构与调色板：两者过拟合的原因不同，混在一起看不出是哪边在涨
        val_parts[:] = [tg / cnt, tp / cnt]
        return tot / cnt

    import copy
    ema = copy.deepcopy(model).eval().requires_grad_(False) if a.ema > 0 else None
    best, log = float("inf"), []
    t0 = time.time()
    n = len(train)
    for step in range(a.steps + 1):
        model.train()
        use32 = T32 is not None and torch.rand(()).item() < a.p32
        D, nD, bs = (T32, len(train32), a.batch32) if use32 else (T, n, a.batch)
        idx = torch.randint(0, nD, (bs,))
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            loss, parts = training_loss(model, *batch_of(D, idx, True), pal_smooth=a.pal_smooth)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if ema is not None:
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(a.ema).add_(pm.detach(), alpha=1 - a.ema)
        if step % a.eval_every == 0:
            model.eval()
            em = ema if ema is not None else model           # 采样用 EMA 权重，就用它来选检查点
            vl = val_loss(V, len(val), em)                   # 选检查点只看 16px val
            vg, vp = val_parts
            v32 = val_loss(V32, len(val32), em) if V32 is not None else None
            rec = {"step": step, "train": loss.item(), **parts, "val": vl, "val_grid": vg, "val_pal": vp, "val32": v32,
                   "lr": sched.get_last_lr()[0], "min": (time.time() - t0) / 60}
            log.append(rec)
            print(json.dumps(rec), flush=True)
            json.dump(log, open(a.out / "log.json", "w"), indent=0)
            state = {"model": em.state_dict(), "args": cfg, "step": step, "val": vl,
                     "is_ema": ema is not None}
            torch.save(state, a.out / "last.pt")
            if step in a.save_at:
                torch.save(state, a.out / f"step_{step}.pt")
            if vl < best:
                best = vl
                torch.save(state, a.out / "best.pt")

    # 训练末：用最佳检查点给 val 材质各采一张，存图便于目视
    ck = torch.load(a.out / "best.pt", map_location=dev)
    model.load_state_dict(ck["model"])
    model.eval()
    from PIL import Image
    del T, V, T32, V32, opt
    torch.cuda.empty_cache()
    for size, pool in ((16, val), (32, val32)):
        if not pool:
            continue
        sv = pool[:32]                                    # 共享卡：别在收尾时 OOM（v3/v4 都栽过）
        V2 = to_tensors(sv, cb, a.codes, tindex, temb)
        try:
            pal, grid = sample(model, V2["text"].to(dev), V2["k"].to(dev), n=size,
                               steps=24 if not a.smoke else 4)
        except torch.OutOfMemoryError:
            print(f"样例图 {size}px 显存不足，跳过（检查点已存）", flush=True)
            continue
        imgs = decode(pal, grid, cb)
        rows = len(imgs) // 8
        sheet = np.concatenate([np.concatenate(list(imgs[r * 8:(r + 1) * 8]), 1) for r in range(rows)], 0)
        up = 64 // size
        name = "val_samples.png" if size == 16 else f"val_samples_{size}.png"
        Image.fromarray(sheet).resize((sheet.shape[1] * up, sheet.shape[0] * up),
                                      Image.NEAREST).save(a.out / name)
    print(f"最佳 val {best:.4f} @ step {ck['step']}；样例 -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
