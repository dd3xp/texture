"""大模型两两评审：方法 A vs 方法 B（用户 API 里的大模型，凭据只从环境变量读）。

按本项目吃过亏的协议（`GOAL.md` 要求；教训见 docs/cron_prompt_old 与记忆）：

1. **正反两问去偏**：每对问两次（A 在左 / A 在右），答案不一致即弃用。
2. **先量可解率，再看胜负**（`pilot` 子命令）：取前 N 对真题 + M 对**空对照**（两边放同一张图），
   **只记录每对是否两序一致，不记录谁赢**（字段白名单强制）。
   空对照的一致率就是掷硬币的地板——理论上限 2q(1-q) ≤ 0.5。
   **真题可解率须 ≥ `--min_rate`（默认 0.65）且明显高于空对照**，才允许跑 `full`。
   本项目的教训：76% 判不上来的比较，单批显著全是子集噪声。
3. `full`：全部对，报 A 胜率、二项 p、Jeffreys 区间、弃用数。判官压缩效应：
   正结果作下界；负结果只能说"没测到大效应"。

展示：每张 16px 瓦片最近邻放大到 192，并排再附 3×3 平铺（纹理是平铺使用的）。

    VLM_BASE_URL=... VLM_API_KEY=... python eval/judge_pairs.py pilot --a TRD_x --b B1
    ... full --a TRD_x --b B1
    ... pilot --a TRD_x --b B3 --subset eval/sdpixl_subset.json
"""
import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "analysis/annotate"))
from exact import binom_test, jeffreys          # noqa: E402

Q = ("下面是两张 {n}x{n} 的像素画材质贴图，材质是「{label}」。每张图下方附了它 3x3 平铺的样子。\n"
     "哪一张更像这个材质、更像一张能用的游戏贴图？只回答 A 或 B，不要解释。\n"
     "（第一张是 A，第二张是 B）")
PILOT_FIELDS = {"pair", "kind", "material", "answered", "resolved"}     # 白名单：不许有胜负


def panel(tile):
    t = np.asarray(tile, np.uint8)
    one = Image.fromarray(t).resize((192, 192), Image.NEAREST)
    tiled = Image.fromarray(np.tile(t, (3, 3, 1))).resize((192, 192), Image.NEAREST)
    im = Image.new("RGB", (192, 392), (255, 255, 255))
    im.paste(one, (0, 0))
    im.paste(tiled, (0, 200))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def first_tile(d: Path, slug):
    for name in (f"{slug}_0.png", f"{slug}.png"):
        if (d / name).exists():
            return np.asarray(Image.open(d / name).convert("RGB"))
    return None


def ask_pair(ask, model, label, n, ta, tb, base, key):
    """返回 'A'/'B'（两序一致时）、'inconsistent'、或 None（API 失败）。"""
    q = Q.format(n=n, label=label)
    pa, pb = panel(ta), panel(tb)
    o1, o2 = ask(model, q, [pa, pb], base, key), ask(model, q, [pb, pa], base, key)
    if not o1 or not o2:
        return None
    v1 = "A" if o1.strip().upper().startswith("A") else "B"
    v2 = "B" if o2.strip().upper().startswith("A") else "A"
    return v1 if v1 == v2 else "inconsistent"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot", "full"])
    ap.add_argument("--a", required=True, help="方法 A 目录名（experiments/baselines/<A>/<size>）")
    ap.add_argument("--b", required=True)
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--set", default="E_mat")
    ap.add_argument("--subset", type=Path, default=None, help="如 eval/sdpixl_subset.json")
    ap.add_argument("--n_pilot", type=int, default=15)
    ap.add_argument("--n_null", type=int, default=5)
    ap.add_argument("--min_rate", type=float, default=0.65)
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--root", type=Path, default=ROOT / "experiments/baselines")
    a = ap.parse_args()
    from vlm_judge import ask
    base, key = os.environ.get("VLM_BASE_URL"), os.environ.get("VLM_API_KEY")
    if not base or not key:
        raise SystemExit("需要环境变量 VLM_BASE_URL 与 VLM_API_KEY")

    items = (json.loads(a.subset.read_text(encoding="utf-8"))["items"] if a.subset
             else json.loads((ROOT / "eval/prompt_sets.json").read_text(encoding="utf-8"))[a.set])
    da, db = a.root / a.a / str(a.size), a.root / a.b / str(a.size)
    pairs = []
    for e in items:
        slug = e["material"].rsplit(".", 1)[0]
        ta, tb = first_tile(da, slug), first_tile(db, slug)
        if ta is not None and tb is not None:
            pairs.append((e["prompt"], ta, tb))
    tag = f"{a.a}_vs_{a.b}_{a.size}" + (f"_{a.subset.stem}" if a.subset else "")
    print(f"{a.a} vs {a.b}：可比对 {len(pairs)}")

    if a.cmd == "pilot":
        rng = np.random.default_rng(0)
        order = rng.permutation(len(pairs))
        recs = []
        for i in order[:a.n_pilot]:
            label, ta, tb = pairs[i]
            v = ask_pair(ask, a.model, label, a.size, ta, tb, base, key)
            recs.append({"pair": int(i), "kind": "real", "material": label,
                         "answered": v is not None, "resolved": v in ("A", "B")})
        for i in order[:a.n_null]:
            label, ta, _ = pairs[i]
            v = ask_pair(ask, a.model, label, a.size, ta, ta, base, key)   # 两边同一张图
            recs.append({"pair": int(i), "kind": "null", "material": label,
                         "answered": v is not None, "resolved": v in ("A", "B")})
        for r in recs:
            assert set(r) <= PILOT_FIELDS, f"试点记录里混进了白名单外的字段：{set(r) - PILOT_FIELDS}"
        real = [r for r in recs if r["kind"] == "real" and r["answered"]]
        null = [r for r in recs if r["kind"] == "null" and r["answered"]]
        rr = np.mean([r["resolved"] for r in real]) if real else float("nan")
        nr = np.mean([r["resolved"] for r in null]) if null else float("nan")
        ok = bool(real) and rr >= a.min_rate and (not null or rr > nr)
        out = {"tag": tag, "real_resolved": float(rr), "null_resolved": float(nr),
               "n_real": len(real), "n_null": len(null), "min_rate": a.min_rate, "pass": ok,
               "records": recs}
        p = ROOT / f"experiments/judge_pilot_{tag}.json"
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"可解率：真题 {rr:.0%}（n={len(real)}）  空对照 {nr:.0%}（n={len(null)}）  "
              f"门槛 {a.min_rate:.0%} -> {'**通过**，可以跑 full' if ok else '**不通过**：判官在这个比较上没分辨力，不跑 full'}")
        return 0 if ok else 1

    # full：先确认试点通过
    pp = ROOT / f"experiments/judge_pilot_{tag}.json"
    if not pp.exists() or not json.loads(pp.read_text(encoding="utf-8"))["pass"]:
        raise SystemExit(f"没有通过的试点记录 {pp.name}：先跑 pilot（判官可能分辨不了这个比较）")
    wins = tot = inc = fail = 0
    recs = []
    for i, (label, ta, tb) in enumerate(pairs):
        v = ask_pair(ask, a.model, label, a.size, ta, tb, base, key)
        recs.append({"pair": i, "material": label, "verdict": v})
        if v is None:
            fail += 1
        elif v == "inconsistent":
            inc += 1
        else:
            tot += 1
            wins += v == "A"
    p = binom_test(wins, tot) if tot else float("nan")
    lo, hi = jeffreys(wins, tot) if tot else (float("nan"), float("nan"))
    out = {"tag": tag, "a_wins": wins, "decided": tot, "inconsistent": inc, "api_fail": fail,
           "rate": wins / tot if tot else float("nan"), "p": p, "jeffreys": [lo, hi], "records": recs}
    (ROOT / f"experiments/judge_full_{tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{a.a} 胜 {wins}/{tot} = {wins / max(tot, 1):.0%}  p={p:.3g}  [{lo:.0%},{hi:.0%}]"
          f"   （两序不一致弃 {inc}，API 失败 {fail}；有效 n / 总 n = {tot}/{len(pairs)}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
