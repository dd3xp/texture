"""A4：先验+填充 vs 降采样基线 的分层盲比任务。

与 D6 那轮的关键差别：

- D6 比的是**纯模型**，已确认输给基线（60:40，p=0.032）。
  这里比的是**先验+填充的组合**，是另一个东西。
- 组合只对 42% 的材质加种子（A3k）。若不分层随机抽样，
  57% 无种子的材质会把信号稀释成"没差别"。
  **因此按"有无周期结构"分层**：
  加种子那一层采足（效应只可能出现在那里），
  无种子那一层作为对照（应当与之前无异）。

问题措辞沿用前两轮的"哪一张更像这个材质"，三轮数据可比。
"""

import argparse
import base64
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "metric"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "premise"))
from model import build_model                                    # noqa: E402
from structural_prior import (learn_prior, learn_border, learn_edges,  # noqa: E402
                              make_seed, add_border_union, fill_from_seed)
from build_testset import match_stats                            # noqa: E402
from materials import prompt_for                                 # noqa: E402


def b64(a: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(a.astype(np.uint8)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("experiments/model/hybrid2/best.pt"))
    ap.add_argument("--data", type=Path, default=Path("data/tiles/dataset_k16.json"))
    ap.add_argument("--pairs", type=Path, default=Path("data/contentdb/pairs.json"))
    ap.add_argument("--n-seeded", type=int, default=40, help="有周期结构的材质对数")
    ap.add_argument("--n-plain", type=int, default=20, help="无周期结构的对照对数")
    ap.add_argument("--n-artist", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("experiments/annotate/study_prior.html"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    kw = dict(k=ck["k"], n_materials=ck["n_materials"], size=16,
              d=a["dim"], depth=a["depth"], drop=0.0)
    if ck["arch"] == "hybrid":
        kw["attn_every"] = a.get("attn_every", 1)
    net = build_model(ck["arch"], **kw).to(dev).eval()
    net.load_state_dict(ck["state"])

    ds = json.loads(args.data.read_text())
    pairs = json.loads(args.pairs.read_text())
    bymat, ref = {}, {}
    for s in ds["samples"]:
        if s["size"] != 16:
            continue
        bymat.setdefault(s["material"], []).append(s)
        if s["split"] == "test" and s["material"] not in ref:
            ref[s["material"]] = s

    cands = [m for m in ref if m in pairs and m in ck["mat2id"]
             and len([x for x in bymat[m] if x["split"] == "train"]) >= 4]

    # 先算先验，据此分层
    seeded, plain = [], []
    priors = {}
    for m in cands:
        tr = [s for s in bymat[m] if s["split"] == "train"]
        tiles = [np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16)
                 for s in tr]
        pr = learn_prior(tiles, [len(s["palette"]) for s in tr], material=m)
        bd = learn_border(tiles)
        eg = learn_edges(tiles)
        priors[m] = (pr, bd, eg)
        # 周期 / 整圈边框 / 分边边框，任一命中都算 seeded
        has_edge = any(v.get("active") for v in eg.values())
        (seeded if (pr["rows"] or bd["has_border"] or has_edge) else plain).append(m)
    print(f"可比材质 {len(cands)}：有结构先验 {len(seeded)}，无先验 {len(plain)}")

    rng = random.Random(args.seed)
    rng.shuffle(seeded)
    rng.shuffle(plain)
    picks = ([(m, "seeded") for m in seeded[: args.n_seeded]] +
             [(m, "plain") for m in plain[: args.n_plain]])

    items = []
    sides: dict[tuple, list] = {}
    for m, stratum in picks:
        s = ref[m]
        pal = np.array(s["palette"], np.uint8)
        nk = len(pal)
        art = np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16)
        art_rgb = pal[art]

        pr, bd, eg = priors[m]
        sd = add_border_union(
            make_seed(pr, nk, rng=np.random.default_rng(rng.randrange(10 ** 6))),
            bd, eg, nk)
        torch.manual_seed(rng.randrange(10 ** 6))
        gen = fill_from_seed(net, sd, pal, nk, ck["mat2id"][m], device=dev)
        model_rgb = pal[np.clip(gen, 0, nk - 1)]

        base_rgb = None
        for p in pairs[m]["high"].values():
            try:
                tex = Image.open(p).convert("RGB")
            except Exception:
                continue
            x = np.asarray(tex.resize((16, 16), Image.BOX), float)
            x = match_stats(x, art_rgb.astype(float))
            d = ((x.reshape(-1, 1, 3) - pal.astype(float).reshape(1, -1, 3)) ** 2).sum(-1)
            base_rgb = pal[d.argmin(1).reshape(16, 16)]
            break
        if base_rgb is None:
            continue

        imgs = {"model": model_rgb, "baseline": base_rgb, "artist": art_rgb}
        duels = [("model", "baseline")]
        if len(items) < args.n_artist * 3 and rng.random() < 0.2:
            duels.append(("model", "artist"))
        for x, y in duels:
            # 左右**确定性对半**，不用每题抛硬币。
            # 抛硬币在 60 对上实测偏到 38:25（约 2σ），
            # 标注者若有左侧偏好，这一偏差直接送给模型一份便宜。
            k = sides.setdefault((x, y), [0, 0])
            put_first = k[0] <= k[1]
            k[0 if put_first else 1] += 1
            l, r = (x, y) if put_first else (y, x)
            items.append({"material": m, "label": prompt_for(m), "kind": "real",
                          "struct": round(max(pr["score"], abs(bd["gap"]) / 2,
                                             max((abs(v["gap"]) / 2 for v in eg.values()
                                                  if v.get("active")), default=0.0)), 4),
                          "stratum": stratum, "left": l, "right": r,
                          "limg": b64(imgs[l]), "rimg": b64(imgs[r])})

        if rng.random() < 0.06:
            blur = np.stack([ndimage.gaussian_filter(art_rgb[..., c].astype(float), 3.0)
                             for c in range(3)], -1)
            g = {"good": art_rgb, "blur": blur}
            k = sides.setdefault(("good", "blur"), [0, 0])
            first = k[0] <= k[1]
            k[0 if first else 1] += 1
            l, r = ("good", "blur") if first else ("blur", "good")
            items.append({"material": m, "label": prompt_for(m), "kind": "check",
                          "struct": round(max(pr["score"], abs(bd["gap"]) / 2,
                                             max((abs(v["gap"]) / 2 for v in eg.values()
                                                  if v.get("active")), default=0.0)), 4),
                          "stratum": stratum, "left": l, "right": r,
                          "limg": b64(g[l]), "rimg": b64(g[r])})

    rng.shuffle(items)
    n_seed = sum(1 for i in items if i["stratum"] == "seeded" and i["kind"] == "real")
    n_plain = sum(1 for i in items if i["stratum"] == "plain" and i["kind"] == "real")
    n_chk = sum(1 for i in items if i["kind"] == "check")
    print(f"生成 {len(items)} 对：有种子 {n_seed}，无种子对照 {n_plain}，注意力检查 {n_chk}")
    for (x, y), k in sorted(sides.items()):
        print(f"  左右平衡 {x} vs {y}: {x} 在左 {k[0]} 次，{y} 在左 {k[1]} 次")

    tpl = (Path(__file__).parent / "task_template.html").read_text(encoding="utf-8")
    args.out.write_text(tpl.replace("__ITEMS__", json.dumps(items, ensure_ascii=False)),
                        encoding="utf-8")
    print(f"写入 {args.out}  ({args.out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
