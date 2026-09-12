"""验证集指标的噪声下限：同一个配置、只换采样，指标能抖多少。

上一轮实测（docs/arch_progress.md 2026-09-11 那节）：同一配置只把 `--seed 0` 换成 1，
V_mat 上 KID 5.23 -> 11.97、FID 61.9 -> 71.2。这些抖动比本项目一路用来挑配置的差还大，
所以在拿验证集比配置之前，必须先知道"多大的差才算数"。

做法：一个配置出每材质 K 张 -> 特征只算一次 -> 反复随机抽 m 张/材质（m=1,2,4,K）重算指标。
每个 m 下的标准差就是那个口径的噪声下限。抽样只在已生成的样本里做，所以量到的是
**采样噪声**（模型随机性 + 有限材质），不含训练随机性。

判据（写死，供之后每一轮引用）：两个配置在验证集上的差要大于
`2 * sqrt(sd_a^2 + sd_b^2)`（同一个 m 口径）才算可判读，否则一律记"分不出"。

`--rerank` 改量重排操作点（`eval/rerank.py` 的口径：CLIP **B/16** 打分挑一张，评测用 B/32）：
此时 m = 重排预算，评测集始终是每材质 1 张。项目里唯一撬得动 CLIP 的杠杆一直没有误差棒，
这里补上。

用法：
  python eval/gen_trd.py ... --n 8 --tag nf_<配置名>        # 每材质 8 张
  python eval/noise_floor.py --set V_mat --methods nf_a nf_b
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "model"))
from metrics import (clip_image_emb, clip_text_emb, dino_feats,      # noqa: E402
                     inception_feats, kid, prompt_of, _sqrtm_trace)
from prompts import is_material, load_set                            # noqa: E402
from tiles_data import load                                          # noqa: E402


class Frechet:
    """固定参照集的 Fréchet 距离；参照那一侧的矩阵平方根只算一次（tr sqrt(AB) = tr sqrt(BA)）。"""

    def __init__(self, ref):
        self.mu = ref.mean(0)
        self.cov = np.cov(ref, rowvar=False)
        w, v = np.linalg.eigh(self.cov)
        self.sqrt = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
        self.tr = np.trace(self.cov)

    def __call__(self, gen):
        mu, cov = gen.mean(0), np.cov(gen, rowvar=False)
        w = np.linalg.eigvalsh(self.sqrt @ cov @ self.sqrt)
        cross = float(np.sqrt(np.clip(w, 0, None)).sum())
        return float(((self.mu - mu) ** 2).sum() + self.tr + np.trace(cov) - 2 * cross)


def load_groups(d: Path, slugs, k):
    """每材质取前 k 张；有材质不足 k 张就退 2（口径不齐的表不能比）。"""
    groups = []
    for s in slugs:
        many = [p for p in d.glob(f"{s}_*.png") if re.fullmatch(re.escape(s) + r"_\d+", p.stem)]
        many.sort(key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        if len(many) < k:
            raise SystemExit(f"[退 2] {d} 的 {s} 只有 {len(many)} 张，要求每材质 >= {k} 张")
        groups.append([np.asarray(Image.open(p).convert("RGB")) for p in many[:k]])
    return groups


def draws(n_mat, k, m, n_draws, seed, score=None):
    """每次抽样返回一个下标数组（每个材质抽 m 张，不放回）。m == k 时只有一种取法。

    给了 score（形状 [n_mat, k] 的 B/16 图文相似度）就只留每材质分最高的那张 = 重排操作点。"""
    base = np.arange(n_mat)[:, None] * k
    rng = np.random.default_rng(seed)
    picks = ([np.tile(np.arange(k), (n_mat, 1))] if m >= k else
             [np.stack([rng.choice(k, m, replace=False) for _ in range(n_mat)])
              for _ in range(n_draws)])
    if score is not None:
        picks = [p[np.arange(n_mat), np.take_along_axis(score, p, 1).argmax(1)][:, None] for p in picks]
    return [(base + p).ravel() for p in picks]


def clip16_scores(groups, prompts, weights):
    """每张瓦片对自己材质名的 CLIP B/16 相似度 -> [材质数, k]（重排用的打分模型，与评测的 B/32 相互独立）。"""
    import torch
    import torch.nn.functional as F
    from transformers import CLIPModel, CLIPTokenizer
    from metrics import DEV, batched, upscale
    m = CLIPModel.from_pretrained(weights).to(DEV).eval()
    tok = CLIPTokenizer.from_pretrained(weights)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    flat = [t for g in groups for t in g]
    x = (upscale(flat, 224) - mean) / std
    ie = F.normalize(batched(lambda b: m.get_image_features(pixel_values=b), x), dim=-1)
    txt = tok([prompt_of(e["prompt"]) for e in prompts], padding=True, return_tensors="pt").to(DEV)
    with torch.no_grad():
        te = F.normalize(m.get_text_features(**txt).float().cpu(), dim=-1)
    k = len(groups[0])
    return (ie.view(len(groups), k, -1) * te[:, None, :]).sum(-1).numpy()


def stats(vals):
    v = np.asarray(vals, float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()), "draws": len(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="V_mat")
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--root", type=Path, default=ROOT / "experiments/baselines")
    ap.add_argument("--k", type=int, default=8, help="每材质已生成的样本数")
    ap.add_argument("--subsets", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--n_draws", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rerank", action="store_true",
                    help="m 改成重排预算：每材质抽 m 张、按 CLIP B/16 挑 1 张（eval/rerank.py 同口径）")
    ap.add_argument("--clip16", default=str(ROOT / "weights/clip-vit-base-patch16"))
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/noise_floor.json")
    a = ap.parse_args()

    prompts, ref_split = load_set(a.set)
    slugs = [e["material"].rsplit(".", 1)[0] for e in prompts]
    ref_tiles = [s["palette"][s["idx"]] for s in load(a.size, ref_split)
                 if a.set.endswith("_all") or is_material(s["material"])]
    if len(ref_tiles) < 50:
        raise SystemExit(f"[退 2] 参照只有 {len(ref_tiles)} 张，分布指标无意义")
    print(f"{a.set}: {len(prompts)} 个材质，每材质 {a.k} 张；参照 {len(ref_tiles)} 张真人 {a.size}px",
          flush=True)
    ref_inc, ref_dino = inception_feats(ref_tiles), dino_feats(ref_tiles)
    fid, fdd = Frechet(ref_inc), Frechet(ref_dino)

    res = {}
    for meth in a.methods:
        groups = load_groups(a.root / meth / str(a.size), slugs, a.k)
        flat = [t for g in groups for t in g]
        inc, dino = inception_feats(flat), dino_feats(flat)
        ie = clip_image_emb(flat)
        te = clip_text_emb([prompt_of(e["prompt"]) for e in prompts for _ in range(a.k)])
        clip_per = (100 * (ie * te).sum(-1).clamp(min=0)).numpy()
        score = clip16_scores(groups, prompts, a.clip16) if a.rerank else None
        res[meth] = {}
        for m in sorted(set(a.subsets + [a.k])):
            acc = {"KID_x1e3": [], "FID": [], "FD_DINOv2": [], "CLIP": []}
            for idx in draws(len(prompts), a.k, m, a.n_draws, a.seed, score):
                acc["KID_x1e3"].append(1e3 * kid(inc[idx], ref_inc)[0])
                acc["FID"].append(fid(inc[idx]))
                acc["FD_DINOv2"].append(fdd(dino[idx]))
                acc["CLIP"].append(float(clip_per[idx].mean()))
            res[meth][m] = {k: stats(v) for k, v in acc.items()}
            print(f"  {meth:<24} m={m:<2} " + "  ".join(
                f"{k}={s['mean']:.2f}+-{s['sd']:.2f}" for k, s in res[meth][m].items()), flush=True)

    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("\n| 方法 | m | KID x1e3 | FID | FD-DINOv2 | CLIP |")
    print("|---|---|---|---|---|---|")
    for meth, per_m in res.items():
        for m, r in per_m.items():
            print(f"| {meth} | {m} | " + " | ".join(
                f"{r[c]['mean']:.2f} +- {r[c]['sd']:.2f}" for c in
                ("KID_x1e3", "FID", "FD_DINOv2", "CLIP")) + " |")

    if len(a.methods) >= 2:
        print("\n判据：|差| > 2*sqrt(sd_a^2+sd_b^2) 才算可判读")
        for i, ma in enumerate(a.methods):
            for mb in a.methods[i + 1:]:
                for m in sorted(set(a.subsets + [a.k])):
                    if m >= a.k:
                        continue                      # m=k 只有一种取法，没有 sd
                    print(f"  {ma} vs {mb}  m={m}")
                    for c in ("KID_x1e3", "FID", "FD_DINOv2", "CLIP"):
                        x, y = res[ma][m][c], res[mb][m][c]
                        d = x["mean"] - y["mean"]
                        thr = 2 * np.hypot(x["sd"], y["sd"])
                        ok = "可判读" if abs(d) > thr else "分不出"
                        print(f"    {c:<10} 差 {d:+.2f}  门槛 {thr:.2f}  -> {ok}")
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
