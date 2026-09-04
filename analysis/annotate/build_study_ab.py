"""B2：真人 vs 降采样基线 的盲比。

这是项目里一直缺的一格。已知：
  - 基线 vs 我们的模型：基线赢 83%（A4）
  - 真人 vs 我们的模型：真人赢 6:1（A4）
  - **真人 vs 基线：从未测过**

它决定还有没有研究空间：
  基线若已接近真人 -> 直接用基线，项目收尾；
  真人若明显赢 -> 那个差距就是唯一值得做的东西，且能从图上看出差在哪。

小规模、快：24 对真人/基线 + 3 对注意力检查，约 3 分钟。
按结构分分层，两层各半，这样能同时回答"差距是否集中在几何结构材质上"。
"""

import argparse
import base64
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "metric"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "premise"))
from structural_prior import learn_prior                           # noqa: E402
from build_testset import match_stats                              # noqa: E402
from materials import prompt_for                                   # noqa: E402


def b64(a):
    buf = io.BytesIO()
    Image.fromarray(a.astype(np.uint8)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--n-check", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path,
                    default=Path("experiments/annotate/study_ab.html"))
    args = ap.parse_args()

    ds = json.loads(Path("data/tiles/dataset_k16.json").read_text())
    pairs = json.loads(Path("data/contentdb/pairs.json").read_text())
    bymat, ref = {}, {}
    for s in ds["samples"]:
        if s["size"] != 16:
            continue
        bymat.setdefault(s["material"], []).append(s)
        if s["split"] == "test" and s["material"] not in ref:
            ref[s["material"]] = s

    cands = []
    for m in sorted(ref):
        tr = [x for x in bymat[m] if x["split"] == "train"]
        if len(tr) < 4 or m not in pairs:
            continue
        raw = [np.frombuffer(bytes.fromhex(x["idx"]), np.uint8).reshape(16, 16)
               for x in tr]
        pr = learn_prior(raw, [len(x["palette"]) for x in tr], material=m)
        cands.append((m, float(pr["score"]), bool(pr["rows"])))
    struct = [c for c in cands if c[2]]
    plain = [c for c in cands if not c[2]]
    print(f"可比材质 {len(cands)}：有几何结构 {len(struct)}，无 {len(plain)}")

    rng = random.Random(args.seed)
    rng.shuffle(struct)
    rng.shuffle(plain)
    half = args.n // 2
    picks = ([(m, sc, "structured") for m, sc, _ in struct[:half]] +
             [(m, sc, "plain") for m, sc, _ in plain[:args.n - half]])

    items, sides = [], {}
    for m, sc, stratum in picks:
        s = ref[m]
        pal = np.array(s["palette"], np.uint8)
        art = np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16)
        art_rgb = pal[art]
        base_rgb = None
        for p in pairs[m]["high"].values():
            try:
                tex = Image.open(p).convert("RGB")
            except Exception:
                continue
            x = match_stats(np.asarray(tex.resize((16, 16), Image.BOX), float),
                            art_rgb.astype(float))
            d = ((x.reshape(-1, 1, 3) - pal.astype(float).reshape(1, -1, 3)) ** 2).sum(-1)
            base_rgb = pal[d.argmin(1).reshape(16, 16)]
            break
        if base_rgb is None:
            continue
        imgs = {"artist": art_rgb, "baseline": base_rgb}
        k = sides.setdefault(("artist", "baseline"), [0, 0])
        first = k[0] <= k[1]
        k[0 if first else 1] += 1
        l, r = ("artist", "baseline") if first else ("baseline", "artist")
        items.append({"material": m, "label": prompt_for(m), "kind": "real",
                      "struct": round(sc, 4), "stratum": stratum,
                      "left": l, "right": r,
                      "limg": b64(imgs[l]), "rimg": b64(imgs[r])})

    chk_src = [p for p in picks[:args.n_check]]
    for m, sc, stratum in chk_src:
        s = ref[m]
        pal = np.array(s["palette"], np.uint8)
        art_rgb = pal[np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16)]
        blur = np.stack([ndimage.gaussian_filter(art_rgb[..., c].astype(float), 3.0)
                         for c in range(3)], -1)
        g = {"good": art_rgb, "blur": blur}
        k = sides.setdefault(("good", "blur"), [0, 0])
        first = k[0] <= k[1]
        k[0 if first else 1] += 1
        l, r = ("good", "blur") if first else ("blur", "good")
        items.append({"material": m, "label": prompt_for(m), "kind": "check",
                      "struct": round(sc, 4), "stratum": stratum,
                      "left": l, "right": r, "limg": b64(g[l]), "rimg": b64(g[r])})

    rng.shuffle(items)
    n_real = sum(1 for i in items if i["kind"] == "real")
    print(f"生成 {len(items)} 对：真人vs基线 {n_real}，注意力检查 {len(items)-n_real}")
    for (x, y), k in sorted(sides.items()):
        print(f"  左右平衡 {x} vs {y}: {k[0]} / {k[1]}")

    tpl = (Path(__file__).parent / "task_template.html").read_text(encoding="utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tpl.replace("__ITEMS__", json.dumps(items, ensure_ascii=False)),
                        encoding="utf-8")
    print(f"写入 {args.out}  ({args.out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
