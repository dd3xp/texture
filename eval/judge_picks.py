"""判官校准：**记下判官每次挑的是哪个位置**，而不是只记两序一致与否。

为什么要这个（`6c64796` 定下的下一步，`830b2d2` 已排除刺激那条解释）：
`judge_pairs.py:73-78` 把两次回答立刻压成 A / B / inconsistent，**扔掉了"挑的是第一张还是第二张"**。
于是"判出率 51% vs 74%"有两种读法，从已落盘的数据里**永远分不开**，而它们对胜率的修正**方向相反**：

  (M-a) **不敏感带**：判官分不出时**扔硬币**。这时两序不一致的对就是"真的很接近"的对，
        丢掉它们会把剩下的 logit 往外推 —— 这是 (L2) 当初假设的那个偏。
  (M-b) **按位置作答**：判官分不出时**照位置挑**（挑第一张的概率 q），换序就翻面 → 必然不一致。
        这时**判出的那一批里混进了侥幸一致的位置作答**，而它们在方法口径下是 50/50 →
        **观测胜率被往 50% 稀释**，判出率越低稀释越狠。S1（丢平局）不是往外推，而是**推得不够**。

**空对照已经把话说了一半**：两图完全相同时判官按定义没有任何内容信息，(M-a) 预测判出率 50%，
实测却是 **21/118 = 17.8%**（`pilot_gate_audit.py`，p vs 50% = 7.3e-13）→ 判官"分不出"时是
**按位置作答**，不是扔硬币。由 2q(1-q) = 0.178 反推 **q = 0.901**。
但这是在**两图相同**时量的；真题上判官分不出的那部分是否也这样，**没人量过**。这个脚本去量。

模型（两参数，其中 q 是空对照独立量出来的，不是拟合的）：每对以概率 λ 进入"看内容"模式
（两序都挑同一个**方法**），否则进入"按位置"模式（每次以 q 挑第一张，两次独立）。
关键在于**看内容模式对"挑第一张的比例"的贡献恰好是 0.5**——它在正序挑第一张、反序就挑第二张，
与它偏爱哪个方法无关。**于是这个读数不含任何自由参数。**

    p_first = 0.5 λ + q (1 - λ)      λ = (R - f) / (1 - f)，f = 17.8%，R = 该臂已发表的判出率

⚠ **本次不产生任何胜率，也不许登账成一条臂**：它是 136 对的校准样本，不是重跑 e3 / e4。
   已发表的 e3 / e4 判决不受影响。判据与点预测见 `analysis/arch/pick_decomp.py`（跑之前提交）。
⚠ **不碰 `judge_pairs.py`**：`eval/b3_32.sh:31` 还会去调它判 B3@32，改它等于在活件上动刀。
   让今后每条 full 都记下位置，是 B3@32 判完之后的事（已记在账本"下一步"里）。

    VLM_BASE_URL=... VLM_API_KEY=... python eval/judge_picks.py --a B2 --b B2up16 --size 32 --n 136
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "analysis/annotate"))
sys.path.insert(0, str(ROOT / "eval"))
from judge_pairs import Q, first_tile, panel        # noqa: E402  （只读地复用，绝不改它）
from prompts import load_set                        # noqa: E402


def pick_of(out):
    """把一次回答转成它挑了哪个**位置**。None = API 失败或答非所问。"""
    if not out:
        return None
    u = out.strip().upper()
    return "first" if u.startswith("A") else ("second" if u.startswith("B") else None)


def verdict_of(p1, p2):
    """与 `judge_pairs.ask_pair` 完全同一套折叠规则（第二问里 A 与 B 的角色是反的）。

    正序：第一张 = 方法 A。反序：第一张 = 方法 B。
    所以 v1 = A ⟺ p1 = first；v2 = A ⟺ p2 = second。
    """
    if p1 is None or p2 is None:
        return None
    v1 = "A" if p1 == "first" else "B"
    v2 = "A" if p2 == "second" else "B"
    return v1 if v1 == v2 else "inconsistent"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--set", default="E_mat")
    ap.add_argument("--n", type=int, default=136, help="抽多少对（种子固定，各边抽到同一批材质）")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--root", type=Path, default=ROOT / "experiments/baselines")
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/judge_picks"))
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    from vlm_judge import ask
    base, key = os.environ.get("VLM_BASE_URL"), os.environ.get("VLM_API_KEY")
    if not base or not key:
        raise SystemExit("需要环境变量 VLM_BASE_URL 与 VLM_API_KEY")

    items = load_set(a.set)[0]
    da, db = a.root / a.a / str(a.size), a.root / a.b / str(a.size)
    pairs = []
    for e in items:
        slug = e["material"].rsplit(".", 1)[0]
        ta, tb = first_tile(da, slug), first_tile(db, slug)
        if ta is not None and tb is not None:
            pairs.append((e["prompt"], ta, tb))
    idx = np.random.default_rng(a.seed).permutation(len(pairs))[:a.n]
    tag = f"{a.a}_vs_{a.b}_{a.size}"
    print(f"{tag}：可比对 {len(pairs)}，抽 {len(idx)} 对（seed={a.seed}）", flush=True)

    recs, fail = [], 0
    for j, i in enumerate(sorted(int(x) for x in idx)):
        label, ta, tb = pairs[i]
        pa, pb = panel(ta), panel(tb)
        q = Q.format(n=a.size, label=label)
        p1 = pick_of(ask(a.model, q, [pa, pb], base, key))
        p2 = pick_of(ask(a.model, q, [pb, pa], base, key))
        v = verdict_of(p1, p2)
        fail += v is None
        recs.append({"pair": i, "material": label, "pick1": p1, "pick2": p2, "verdict": v})
        if (j + 1) % 20 == 0:
            print(f"  [{j + 1}/{len(idx)}]", flush=True)

    good = [r for r in recs if r["verdict"] is not None]
    n_first = sum((r["pick1"] == "first") + (r["pick2"] == "first") for r in good)
    out = {"tag": tag, "n_pairs": len(recs), "api_fail": fail, "seed": a.seed,
           "n_answers": 2 * len(good), "n_first": n_first,
           "resolved": sum(r["verdict"] in ("A", "B") for r in good), "records": recs}
    (a.outdir / f"judge_picks_{tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{tag}：挑第一张 {n_first}/{2 * len(good)} = {n_first / max(2 * len(good), 1):.1%}"
          f"   判出 {out['resolved']}/{len(good)}   API 失败 {fail}")
    print("（判据与点预测在 analysis/arch/pick_decomp.py，本脚本不作判读、不报胜率）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
