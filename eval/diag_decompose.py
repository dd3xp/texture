"""把 TRD 与真人之间的分布差距拆成"结构"与"配色"两半（验证集，只用于决定下一步改哪里）。

TRD 的表示天然可拆：瓦片 = 亮度序索引网格（结构）+ 调色板（配色）。对 V-mat 每张真人瓦片，
用它的材质名、**同样的色数 k** 让 TRD 出一张，然后两两互换调色板（同 k，亮度序一一对应）：

  TRD           TRD 网格 + TRD 调色板
  结构=真人      真人网格 + TRD 调色板   → 与真人的差距只来自配色
  配色=真人      TRD 网格 + 真人调色板   → 与真人的差距只来自结构
  真人（n 对半） 参照集自身的一半 vs 另一半 → 地板

哪一行离"TRD"更近、离地板更远，差距就主要在哪一半。
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
from trd import sample                                         # noqa: E402
from train_trd import decode, clip_text, TEXT_TMPL, model_from_args   # noqa: E402
from colour_task import targets                                # noqa: E402
from tiles_data import load, canonicalise                      # noqa: E402
from metrics import evaluate                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ckpt", default="last.pt")
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--size", type=int, default=16,
                    help="(M53) 画布尺寸。默认 16 ＝ (M46)–(M52) 那条路径一个字未改；"
                         "给 32 时参照集换成 V_mat 的 32px 真人瓦片、网格按 32 采样")
    ap.add_argument("--bs", type=int, default=32,
                    help="(M53) 生成批大小。默认 32 ＝ 旧路径（16px 一个字未改）；"
                         "32px 上 bs=32 会 OOM（实测吃满 44.5GB），按 final_test.sh 的 TRD32 用 8")
    ap.add_argument("--floor_reps", type=int, default=1,
                    help="(M53) 地板行重复几次随机对半。>1 时额外落盘 `_floor_null`＝"
                         "「真人 vs 真人」在**这个 n 上**的经验零分布（(M50) 那条纪律：门槛口径不对就自己造零分布）。"
                         "默认 1 ＝ 旧行为（前一半 vs 后一半），逐位不变")
    ap.add_argument("--reps", type=int, default=2, help="每个目标出几张（增加 n 降方差）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--xmodal", action="store_true",
                    help="再加一行 palette=xmodal：正式配置那套跨模态检索（B/16 图文相似度，n_name=30，前 5 随机）")
    ap.add_argument("--oracle_split", action="store_true",
                    help="再加三行 oracle 重排（只用于把「挑调色板」这一轴再拆一级，需 --xmodal）："
                         "在前 5 / 前 30 候选 / 全库里按「与该目标真人调色板的距离」取最优。"
                         "⚠ 全部用到目标信息，是不可达上界，只作定位、不是可达读数")
    ap.add_argument("--oracle_feat", action="store_true",
                    help="(M48) 再加四行 oracle 重排，但**在评测特征空间里选**（Inception pool3，"
                         "= KID/FID 用的同一张网）：前 5 / 前 30 / 全库随机 30 / 全库随机 N 里，"
                         "按 ||Inception(候选调色板配这张网格) - Inception(该目标真人瓦片)||^2 取最优。"
                         "需 --xmodal。⚠ 用到目标信息，是不可达上界，只作定位")
    ap.add_argument("--rand_pool", type=int, default=512, help="--oracle_feat 的全库随机候选池大小")
    ap.add_argument("--rerank", action="store_true",
                    help="(M49) 再加四行：在**正式配置那 5 个候选**里换个挑法。三行是**可达**的"
                         "（看不到目标）：rr_argmax 取图文相似度最高的一张、rr_trdpal 取与模型自己"
                         "预测的调色板最近的一张、rr_incTRD 在 Inception pool3 里取与模型自己那张"
                         "瓦片最近的一张；外加 incbest5（用目标特征选，(M48) 的不可达上界）作锚点。"
                         "需 --xmodal，与 --oracle_feat 互斥")
    ap.add_argument("--sham", type=int, default=0,
                    help="(M50) 再加 N 行 palette=shamJ：在**同一 5 张候选**里独立均匀重抽一张。"
                         "正式配置那张也是均匀随机 ⇒ 两者可交换 ⇒ 这 N 行的 G 真值恒为 0，"
                         "就是 G 的**经验零分布**（于是 rr_* 对它的秩检验是精确置换检验）。"
                         "抽样用独立 rng、且在生成循环之后 ⇒ ⛔ 不消耗生成流，原有各行逐位复现。需 --rerank")
    ap.add_argument("--out", type=Path, default=None, help="落盘路径（/mnt/data 常年贴满，跑远程时指到 /tmp）")
    ap.add_argument("--dump", type=Path, default=None,
                    help="(M52) 把各行的瓦片按判官要的目录结构落盘：<dump>/<行名>/<size>/<slug>_0.png，"
                         "于是 `judge_pairs.py --root <dump> --set V_mat` 能直接拿两行去对判。"
                         "只落 reps 的第一轮、每个 slug 的 j=0 那张。⚠ 在生成循环之后做，不消耗任何 rng")
    a = ap.parse_args()
    if a.rerank and (a.oracle_feat or not a.xmodal):
        ap.error("--rerank 需要 --xmodal，且与 --oracle_feat 互斥（行名会撞）")
    if a.sham and not a.rerank:
        ap.error("--sham 需要 --rerank（零分布要落在同一个 5 张候选集上）")
    dev = "cuda"
    torch.manual_seed(a.seed)
    ck = torch.load(a.run / a.ckpt, map_location=dev)
    cb = np.load(a.run / "codebook.npy")
    model = model_from_args(ck["args"], drop=0.0).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    T = targets("V_mat", a.size)
    ref = [t["ref"] for t in T]
    # 真人：亮度序网格 + 调色板（与 tiles_data 同一规范化）
    real = []
    for t in T:
        cols, inv = np.unique(t["ref"].reshape(-1, 3), axis=0, return_inverse=True)
        g, p = canonicalise(inv.reshape(a.size, a.size).astype(np.int64), cols.astype(np.uint8))
        real.append((g, p))
    temb = clip_text([TEXT_TMPL.format(p=t["prompt"]) for t in T], dev).to(dev)
    ks = torch.tensor([p.shape[0] for _, p in real], device=dev).clamp(max=16)

    # 检索调色板：训练集（去污染）里同色数、材质名 CLIP 文本嵌入最近的前 5 张之一的调色板
    train = load(16, "train", extra=True)
    tr_m = sorted({s["material"] for s in train})
    from train_trd import text_prompt
    tr_e = clip_text([text_prompt(m) for m in tr_m], dev).to(dev)
    tr_ix = {m: i for i, m in enumerate(tr_m)}
    by_k = {}
    for s in train:
        by_k.setdefault(s["k_used"], []).append(s)
    rng = np.random.default_rng(a.seed)

    def retrieve(i, k):
        pool = by_k.get(k) or by_k[min(by_k, key=lambda kk: abs(kk - k))]
        sims = (tr_e[[tr_ix[s["material"]] for s in pool]] @ temb[i]).cpu().numpy()
        top = np.argsort(-sims)[:5]
        return pool[top[rng.integers(len(top))]]["palette"]

    # 正式配置（final_test.sh 的 TRD16）用的是跨模态检索：先按名字取 n_name=30 条，
    # 再按"这张真人瓦片 ↔ 材质名"的 CLIP-B/16 图文相似度取前 5 随机一张，且不限色数（FREE_K）。
    # 这里只把它当**调色板来源**加成一行，网格仍是同一张 —— 四行共用一张网格，隔离出"挑调色板"这一轴。
    def pal_rs(p, n=16):
        """调色板重采样到 n 个亮度槽（两张调色板色数不同也能比）。两侧都已按亮度排序。"""
        p = np.asarray(p, np.float64)
        return p[np.round(np.linspace(0, len(p) - 1, n)).astype(np.int64)]

    mem = t16 = bank_rs = None
    if a.xmodal:
        sys.path.insert(0, str(ROOT / "model"))
        from palette_memory import PaletteMemory, clip16, clip16_texts
        mem = PaletteMemory(dev)
        mem.enable_xmodal(dev)
        m16, tok16 = clip16(dev)
        t16 = clip16_texts([t["prompt"] for t in T], dev, m16, tok16)
        del m16, tok16
        torch.cuda.empty_cache()
        if a.oracle_split:
            bank_rs = np.stack([pal_rs(p) for p in mem.pal])          # [M,16,3]

    def xmodal_cands(i):
        """与 PaletteMemory.query(k=None, colour=None, topk=5, n_name=30) **逐条等价**的内联版本，
        rng 消耗顺序一字不差（先 rng.random(len(sims))、再 rng.integers(5)）——这样才能复现 (M46) 的那一行。
        返回 (选中的调色板, 前 5 候选下标, 前 30 候选下标, 选中者在库里的下标)。"""
        sims = (mem.emb @ temb[i].float()).cpu().numpy()
        sims = sims + 1e-6 * rng.random(len(sims))
        near = np.argsort(-sims)[:30]
        xs = (mem.img16[torch.as_tensor(near, device=mem.img16.device)] @ t16[i].float()).cpu().numpy()
        cand = near[np.argsort(-xs)[:5]]
        pick = int(cand[rng.integers(len(cand))])
        return mem.pal[pick], cand, near, pick

    names = ["TRD", "struct=real", "palette=real", "palette=retrieved"] + (["palette=xmodal"] if a.xmodal else [])
    if a.oracle_split:
        names += ["palette=best5", "palette=best30", "palette=best_bank"]
    RAND_R = f"palette=incbest{a.rand_pool}r"
    feat_names = ["palette=incbest5", "palette=incbest30", "palette=incbest30r", RAND_R]
    if a.oracle_feat:
        names += feat_names
    RR = ["palette=rr_argmax", "palette=rr_trdpal", "palette=rr_incTRD"]
    SHAM = [f"palette=sham{j + 1}" for j in range(a.sham)]
    if a.rerank:
        names += ["palette=incbest5"] + RR + SHAM
    rows = {k: [] for k in names}
    jobs = []
    mats = []
    for r in range(a.reps):
        for i in range(0, len(T), a.bs):
            sl = slice(i, i + a.bs)
            pal, grid = sample(model, temb[sl], ks[sl], n=a.size, cfg=a.cfg)
            tiles = decode(pal, grid, cb)
            for j, t in enumerate(T[sl]):
                gi = grid[j].cpu().numpy()
                tp = cb[pal[j].cpu().numpy()[:int(ks[sl][j])]]              # TRD 调色板（亮度序槽）
                rg, rp = real[i + j]
                rows["TRD"].append(tiles[j])
                rows["struct=real"].append(tp[rg].astype(np.uint8))
                rows["palette=real"].append(rp[np.clip(gi, 0, len(rp) - 1)].astype(np.uint8))
                qp = retrieve(i + j, int(ks[sl][j]))
                rows["palette=retrieved"].append(qp[np.clip(gi, 0, len(qp) - 1)].astype(np.uint8))
                if mem is not None:
                    xp, cand, near, pick = xmodal_cands(i + j)
                    rows["palette=xmodal"].append(xp[np.clip(gi, 0, len(xp) - 1)].astype(np.uint8))
                    if a.oracle_feat:
                        jobs.append((gi.copy(), i + j, cand, near, pick))
                    if a.rerank:
                        jobs.append((gi.copy(), i + j, cand, pick, tiles[j], pal_rs(tp)))
                    if a.oracle_split:
                        rr = pal_rs(rp)
                        for nm, pool_ix in (("palette=best5", cand), ("palette=best30", near)):
                            d = np.sqrt(((bank_rs[pool_ix] - rr) ** 2).sum(-1)).mean(-1)
                            bp = mem.pal[int(pool_ix[int(d.argmin())])]
                            rows[nm].append(bp[np.clip(gi, 0, len(bp) - 1)].astype(np.uint8))
                        d = np.sqrt(((bank_rs - rr) ** 2).sum(-1)).mean(-1)
                        bp = mem.pal[int(d.argmin())]
                        rows["palette=best_bank"].append(bp[np.clip(gi, 0, len(bp) - 1)].astype(np.uint8))
                mats.append(t["prompt"])
    out, cache, sel = {}, None, {}
    if a.oracle_feat:
        # (M48) 在**读数用的那把尺子**（Inception pool3）里选 oracle，而不是 RGB 代理距离。
        # 换调色板只是给同一张网格换取色表，不需要重新生成 => 代价只是候选瓦片的特征前向。
        from metrics import inception_feats, dino_feats
        cache = {"inc": inception_feats(ref), "dino": dino_feats(ref)}
        ref_inc = cache["inc"]
        prng = np.random.default_rng(20260917)           # 与生成用的 rng 分开，且抽样在生成循环之后
        r30 = prng.choice(len(mem.pal), 30, replace=False)
        r512 = prng.choice(len(mem.pal), a.rand_pool, replace=False)
        dsum = {k: 0.0 for k in ["palette=xmodal"] + feat_names}
        uniq = {k: set() for k in feat_names}
        for n_done, (gi, ti, cand, near, pick) in enumerate(jobs):
            pool_ix = {"palette=incbest5": cand, "palette=incbest30": near,
                       "palette=incbest30r": r30, RAND_R: r512}
            uni = np.unique(np.concatenate([cand, near, r30, r512, [pick]])).astype(np.int64)
            cand_tiles = [mem.pal[int(x)][np.clip(gi, 0, len(mem.pal[int(x)]) - 1)].astype(np.uint8)
                          for x in uni]
            f = np.concatenate([inception_feats(cand_tiles[s:s + 192])
                                for s in range(0, len(cand_tiles), 192)])
            d = ((f - ref_inc[ti]) ** 2).sum(1)
            pos = {int(v): p for p, v in enumerate(uni)}
            dsum["palette=xmodal"] += float(d[pos[pick]])
            for nm, ixs in pool_ix.items():
                sub = np.array([pos[int(x)] for x in ixs], np.int64)
                b = int(sub[int(d[sub].argmin())])
                rows[nm].append(cand_tiles[b])
                dsum[nm] += float(d[b])
                uniq[nm].add(int(uni[b]))
            if n_done % 50 == 0:
                print(f"  oracle_feat {n_done}/{len(jobs)}", flush=True)
        sel = {k: {"mean_d": dsum[k] / len(jobs), "uniq": len(uniq[k]) if k in uniq else None}
               for k in dsum}
    rr_diag = {}
    if a.rerank:
        # (M49) 同一个 5 张候选集，只换"怎么挑"。三个可达重排器都看不到目标；
        # incbest5 用目标特征选，只作 (M48) 那个不可达上界的锚点。
        from metrics import inception_feats, dino_feats
        cache = {"inc": inception_feats(ref), "dino": dino_feats(ref)}
        ref_inc = cache["inc"]
        picked = {k: [] for k in ["palette=incbest5"] + RR + SHAM}
        dsel = {k: 0.0 for k in ["palette=xmodal", "palette=incbest5"] + RR + SHAM}
        subset_ok = True
        srng = np.random.default_rng(90500 + a.seed)   # 与生成流分开 => 原有各行逐位复现
        for n_done, (gi, ti, cand, pick, trd_tile, tprs) in enumerate(jobs):
            uni = np.unique(np.concatenate([cand, [pick]])).astype(np.int64)
            cand_tiles = [mem.pal[int(x)][np.clip(gi, 0, len(mem.pal[int(x)]) - 1)].astype(np.uint8)
                          for x in uni]
            f = inception_feats(cand_tiles + [trd_tile])
            fc, ftrd = f[:-1], f[-1]
            d_ref = ((fc - ref_inc[ti]) ** 2).sum(1)
            d_trd = ((fc - ftrd) ** 2).sum(1)
            pos = {int(v): p for p, v in enumerate(uni)}
            sub = np.array([pos[int(x)] for x in cand], np.int64)
            l2 = np.sqrt(((np.stack([pal_rs(mem.pal[int(x)]) for x in cand]) - tprs) ** 2)
                         .sum(-1)).mean(-1)
            sel_b = {"palette=incbest5": int(sub[int(d_ref[sub].argmin())]),      # ⚠ 用目标 = 上界
                     "palette=rr_argmax": pos[int(cand[0])],                      # xs 最高的那张
                     "palette=rr_trdpal": pos[int(cand[int(l2.argmin())])],
                     "palette=rr_incTRD": int(sub[int(d_trd[sub].argmin())])}
            for nm in SHAM:                                   # 均匀重抽 = 与正式配置那张可交换
                sel_b[nm] = pos[int(cand[srng.integers(len(cand))])]
            dsel["palette=xmodal"] += float(d_ref[pos[pick]])
            for nm, b in sel_b.items():
                rows[nm].append(cand_tiles[b])
                picked[nm].append(int(uni[b]))
                dsel[nm] += float(d_ref[b])
                subset_ok = subset_ok and int(uni[b]) in set(int(x) for x in cand)
            if n_done % 50 == 0:
                print(f"  rerank {n_done}/{len(jobs)}", flush=True)
        n = len(jobs)
        orc = picked["palette=incbest5"]
        for nm in RR + SHAM:
            p = picked[nm]
            rr_diag[nm] = {
                "change_rate": sum(int(x != jobs[t][3]) for t, x in enumerate(p)) / n,
                "oracle_hit": sum(int(x == orc[t]) for t, x in enumerate(p)) / n,
                "mean_d": dsel[nm] / n, "uniq": len(set(p))}
        rr_diag["palette=xmodal"] = {"mean_d": dsel["palette=xmodal"] / n}
        rr_diag["palette=incbest5"] = {"mean_d": dsel["palette=incbest5"] / n,
                                       "uniq": len(set(orc))}
        rr_diag["subset_ok"] = subset_ok
    if a.dump:
        # (M52) 判官只认 <root>/<方法>/<size>/<slug>_0.png。各行的 tiles 与 T 同序
        # （生成循环与 jobs 循环都是 for r in reps: for i in T），取前 len(T) 个即第一轮。
        from PIL import Image
        first = [i for i, t in enumerate(T) if t["j"] == 0]
        for name, tiles in rows.items():
            d = a.dump / name.replace("palette=", "pal_").replace("=", "_") / str(a.size)
            d.mkdir(parents=True, exist_ok=True)
            for i in first:
                Image.fromarray(np.asarray(tiles[i], np.uint8)).save(d / f"{T[i]['slug']}_0.png")
        print(f"dump -> {a.dump}（{len(rows)} 行 x {len(first)} 张）", flush=True)
    for name, tiles in rows.items():
        res, cache = evaluate(tiles, mats, ref, ref_cache=cache)
        out[name] = res
        print(f"{name:<14} " + "  ".join(f"{k}={v:.3f}" for k, v in res.items() if k != "n"), flush=True)
    # 地板：真人一半对另一半（不同目标，n 小，只作量级参考）
    half = len(ref) // 2
    res, _ = evaluate(ref[:half], [t["prompt"] for t in T[:half]], ref[half:])
    out["real_half"] = res
    if a.floor_reps > 1:
        # (M53) 地板的经验零分布：同一批真人瓦片随机对半 R−1 次。真值恒为「两边同分布」，
        # 于是这组读数就是 KID 在**这个 n 上**的零分布。独立 rng、且在生成循环之后 => 原有各行逐位复现。
        frng = np.random.default_rng(530000 + a.seed)
        null = []
        for _ in range(a.floor_reps - 1):
            perm = frng.permutation(len(ref))
            A, B = perm[:half], perm[half:]
            r, _ = evaluate([ref[i] for i in A], [T[i]["prompt"] for i in A], [ref[i] for i in B])
            null.append(r)
            print(f"  floor_null {len(null)}/{a.floor_reps - 1} KID={r.get('KID_x1e3', float('nan')):.3f}",
                  flush=True)
        out["_floor_null"] = null
    if sel:
        out["_sel"] = sel
    if rr_diag:
        out["_rr"] = {k: v for k, v in rr_diag.items() if k != "subset_ok"}
        out["_rr_subset_ok"] = rr_diag["subset_ok"]
    p = a.out or ROOT / f"experiments/diag_decompose_{a.run.name}.json"
    p.write_text(json.dumps(out, indent=1))      # 落盘一律挪到打印之前：崩在打印上也不丢数据
    print("real half vs half " + "  ".join(f"{k}={v:.3f}" for k, v in res.items() if k != "n"))
    print("->", p)


if __name__ == "__main__":
    main()
