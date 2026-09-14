"""B3（SD-piXL）12 材质子集：CLIP-B/32 配对分数 + 读判官 JSON + B3 出图耗时。判据写在 eval/b3_subset.sh 头。

CLIP 与 eval/run_eval.py 同口径（metrics.evaluate：提示词模板 prompt_of、100*cos 截 0、每材质第 0 张）。
需要 CLIP 权重（HF 缓存）；`--no_clip` 时只读判官 JSON 与日志，零 GPU 零 API。

    python analysis/arch/b3_subset_stats.py --judge_dir experiments/judge_b3
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
from exact import binom_test          # noqa: E402

ARMS = ["TRD16", "TRD16c", "B2", "B7", "B1", "B4", "B3"]
JUDGED = ["TRD16", "TRD16c", "B2", "B7"]


def first_tile(d, slug):
    for name in (f"{slug}_0.png", f"{slug}.png"):
        if (d / name).exists():
            return np.asarray(Image.open(d / name).convert("RGB"))
    return None


def b3_hours():
    """出图日志里每张的耗时（`[gpuN] <prompt>  ok X.XX h`）；日志不在就返回空。"""
    hrs = {}
    for f in [p for d in (ROOT / "remote_tmp", ROOT / "experiments", Path("/tmp")) for p in d.glob("b3*.txt")]:
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"\[gpu\d+\]\s+(.+?)\s+ok ([\d.]+) h", line)
            if m:
                hrs[m.group(1).strip()] = float(m.group(2))
    return hrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge_dir", type=Path, default=ROOT / "experiments/judge_b3")
    ap.add_argument("--root", type=Path, default=ROOT / "experiments/baselines")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no_clip", action="store_true")
    a = ap.parse_args()
    items = json.loads((ROOT / "eval/sdpixl_subset.json").read_text(encoding="utf-8"))["items"]
    slugs = [e["material"].rsplit(".", 1)[0] for e in items]
    prompts = [e["prompt"] for e in items]
    res = {"n_materials": len(items)}

    if not a.no_clip:
        sys.path.insert(0, str(ROOT / "eval"))
        from metrics import clip_image_emb, clip_text_emb, prompt_of
        te = clip_text_emb([prompt_of(p) for p in prompts])
        scores = {}
        for m in ARMS:
            tiles = [first_tile(a.root / m / "16", s) for s in slugs]
            if any(t is None for t in tiles):
                print(f"  {m}: 缺图 {sum(t is None for t in tiles)}/12，跳过")
                continue
            ie = clip_image_emb(tiles)
            scores[m] = (100 * (ie * te).sum(-1).clamp(min=0)).tolist()
        res["clip"] = {m: {"mean": float(np.mean(v)), "per_material": v} for m, v in scores.items()}
        print("CLIP-B/32（12 材质，每材质第 0 张）：")
        for m, v in scores.items():
            line = f"  {m:7s} {np.mean(v):6.2f}"
            if m != "B3" and "B3" in scores:
                d = np.array(v) - np.array(scores["B3"])
                w, n = int((d > 0).sum()), int((d != 0).sum())
                line += f"   vs B3：高 {w}/{n}  符号检验 p={binom_test(w, n):.3g}  均差 {d.mean():+.2f}"
                res["clip"][m]["vs_B3"] = {"higher": w, "n": n, "p": binom_test(w, n), "mean_diff": float(d.mean())}
            print(line)

    print("判官（A vs B3，两序一致才计）：")
    res["judge"] = {}
    for m in JUDGED:
        tag = f"{m}_vs_B3_16_sdpixl_subset"
        pp, fp = a.judge_dir / f"judge_pilot_{tag}.json", a.judge_dir / f"judge_full_{tag}.json"
        if not pp.exists():
            print(f"  {m:7s} 试点未跑")
            continue
        pil = json.loads(pp.read_text(encoding="utf-8"))
        row = {"pilot_real": pil["real_resolved"], "pilot_null": pil["null_resolved"], "pass": pil["pass"]}
        s = f"  {m:7s} 试点 {pil['real_resolved']:.0%}/空 {pil['null_resolved']:.0%}"
        if pil["pass"] and fp.exists():
            f = json.loads(fp.read_text(encoding="utf-8"))
            row.update(wins=f["a_wins"], decided=f["decided"], p=f["p"], jeffreys=f["jeffreys"])
            s += (f"  -> {m} 胜 {f['a_wins']}/{f['decided']}  p={f['p']:.3g}  "
                  f"[{f['jeffreys'][0]:.0%},{f['jeffreys'][1]:.0%}]  弃 {f['inconsistent']}")
        elif not pil["pass"]:
            s += "  -> 不过门槛，不报胜负"
        print(s)
        res["judge"][m] = row

    hrs = b3_hours()
    got = [hrs[p] for p in prompts if p in hrs]
    if got:
        res["b3_gpu_hours"] = {"n": len(got), "median": statistics.median(got), "min": min(got), "max": max(got),
                               "per_prompt": {p: hrs[p] for p in prompts if p in hrs}}
        print(f"B3 每张耗时（共享卡）：n={len(got)} 中位 {statistics.median(got):.2f} h，"
              f"范围 {min(got):.2f}-{max(got):.2f} h")
    if a.out:
        a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
