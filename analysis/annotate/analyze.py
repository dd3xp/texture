"""分析人工标注：(1) 人的偏好排序 (2) 分类器与人的一致率。

第二项是关键——它决定那把 12.8% 的分类器还能不能用。
一致率接近 50% 意味着尺子和人无关，之前所有基于它的结论都要重新看待。

用与 build_task.py 相同的 seed 重建同一批图，保证打分对象一致。
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "metric"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "premise"))
from compare import load_rgb, features                # noqa: E402
from build_testset import match_stats, palette_from   # noqa: E402
from materials import load as load_materials          # noqa: E402
from learn_material import SmallCNN, HELD_OUT         # noqa: E402


def tail(tex, n, ref, pal):
    x = np.asarray(tex.resize((n, n), Image.BOX), float)
    x = match_stats(x, ref)
    d = ((x.reshape(-1, 1, 3) - pal.reshape(1, -1, 3)) ** 2).sum(-1)
    return pal[d.argmin(1)].reshape(x.shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("experiments/annotate/annotations.csv"))
    ap.add_argument("--pairs", type=Path, default=Path("data/contentdb/pairs.json"))
    ap.add_argument("--testset", type=Path, default=Path("experiments/metric/testset"))
    ap.add_argument("--ckpt", type=Path,
                    default=Path("experiments/metric/material_cnn_gray.pt"))
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--n-materials", type=int, default=22)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.csv.open(encoding="utf-8")))
    real = [r for r in rows if r["kind"] == "real"]
    chk = [r for r in rows if r["kind"] == "check"]
    print(f"标注 {len(rows)} 条：有效比较 {len(real)}，注意力检查 {len(chk)} "
          f"（正确 {sum(1 for r in chk if r['chosen'] == 'good')}/{len(chk)}）\n")

    # ---------- 1. 人的偏好 ----------
    print("== 人的偏好：条件两两对决 ==")
    print(f"{'对决':<34}{'胜':>6}{'负':>6}{'平':>6}{'胜率':>9}{'双尾 p':>10}")
    print("-" * 72)
    duel = defaultdict(lambda: [0, 0, 0])
    for r in real:
        a, b = sorted([r["left"], r["right"]])
        if r["choice"] == "tie":
            duel[(a, b)][2] += 1
        elif r["chosen"] == a:
            duel[(a, b)][0] += 1
        else:
            duel[(a, b)][1] += 1
    for (a, b), (w, l, t) in sorted(duel.items()):
        n = w + l
        p = stats.binomtest(w, n, 0.5).pvalue if n else float("nan")
        rate = w / n if n else float("nan")
        print(f"{a+' vs '+b:<34}{w:>6}{l:>6}{t:>6}{rate:>8.0%}{p:>10.3f}")

    wins = Counter()
    for r in real:
        if r["choice"] != "tie":
            wins[r["chosen"]] += 1
    tot = Counter()
    for r in real:
        tot[r["left"]] += 1
        tot[r["right"]] += 1
    print(f"\n{'条件':<22}{'出场':>7}{'获胜':>7}{'总胜率':>9}")
    print("-" * 46)
    for c in sorted(tot):
        print(f"{c:<22}{tot[c]:>7}{wins[c]:>7}{wins[c]/tot[c]:>8.0%}")

    # ---------- 2. 分类器与人是否一致 ----------
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cls = ck["classes"]
    assert ck.get("gray")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = SmallCNN(len(cls)).to(dev).eval()
    net.load_state_dict(ck["state"])

    def score(a, mat):
        g = (a / 255.0) @ np.array([0.299, 0.587, 0.114], np.float32)
        g = (g - g.mean()) / (g.std() + 1e-6)
        t = torch.tensor(np.repeat(g[..., None], 3, -1).astype(np.float32))
        with torch.no_grad():
            p = net(t.permute(2, 0, 1)[None].to(dev)).softmax(-1)[0].cpu()
        return float(p[cls[mat]])

    # 用同一 seed 重建图，保证与标注者看到的一致
    rng = random.Random(args.seed)
    pairs = json.loads(args.pairs.read_text())
    names = load_materials()
    imgs = {}
    for m in [x for x in names if x in pairs][: args.n_materials]:
        held = [p for pk, p in pairs[m]["low"].items() if pk in HELD_OUT]
        if not held:
            continue
        A = load_rgb(held[0], args.size)
        ref = load_rgb(next(iter(pairs[m]["low"].values())), args.size)
        if A is None or ref is None:
            continue
        pal = palette_from(ref, features(ref)["n_colors"])
        try:
            B = tail(Image.open(next(iter(pairs[m]["high"].values()))).convert("RGB"),
                     args.size, ref, pal)
        except Exception:
            continue
        sd = args.testset / f"sdxl_{m}"
        if not sd.exists():
            sd = args.testset / f"sdxl_{m}.png"
        d = {"A_artist_native": A, "B_pipe_artist": B}
        if sd.exists():
            d["C_pipe_sdxl"] = tail(Image.open(sd).convert("RGB"), args.size, ref, pal)
        imgs[m] = d

    agree = dis = skip = 0
    per_duel = defaultdict(lambda: [0, 0])
    for r in real:
        if r["choice"] == "tie":
            continue
        m = r["material"]
        if m not in imgs or r["left"] not in imgs[m] or r["right"] not in imgs[m]:
            skip += 1
            continue
        sl = score(imgs[m][r["left"]], m)
        sr = score(imgs[m][r["right"]], m)
        pick = r["left"] if sl > sr else r["right"]
        key = " vs ".join(sorted([r["left"], r["right"]]))
        if pick == r["chosen"]:
            agree += 1
            per_duel[key][0] += 1
        else:
            dis += 1
            per_duel[key][1] += 1

    n = agree + dis
    print(f"\n== 分类器与人的一致率 ==")
    if n:
        p = stats.binomtest(agree, n, 0.5).pvalue
        print(f"一致 {agree} / 不一致 {dis}  → {agree/n:.0%}  (双尾 p={p:.3f}, 跳过 {skip})")
        print(f"\n{'对决':<34}{'一致':>7}{'不一致':>8}{'一致率':>9}")
        print("-" * 60)
        for k, (a_, d_) in sorted(per_duel.items()):
            print(f"{k:<34}{a_:>7}{d_:>8}{a_/(a_+d_):>8.0%}")
        print("\n判读：一致率显著高于 50% → 尺子与人相关，可继续使用")
        print("      接近 50% → 尺子与人无关，基于它的结论需重新审视")


if __name__ == "__main__":
    main()
