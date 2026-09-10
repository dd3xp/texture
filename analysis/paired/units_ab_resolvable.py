"""3.2 对 4.5 裁剪盲比的**前置可解率试点**——先量判官能不能分，再决定花不花 GPU 做整轮。

## 为什么先做这个

第 74 轮（`f5cf75c`）确认了单元惯例残差：交付瓦片中位 4.00 对真人 3.20，
MW p=6.88e-08。判读里写着「值得下一轮渲染高分源、按 3.2 与 4.5 各裁一份做盲比」，
并明确「这**不**等于 3.2 更好看」——哪个更好看只能靠盲比回答。

但同一天（第 75 轮，`2ea1f6e`）刚立了一条规矩，就是为了防这一步犯错：
`spread_rescale` 可辨认性试点 39 对里只有 11 对可用（76% 正反不一致），
那 11 对上 10/11、p=0.0117，我据此宣布路线重开；扩到 75 对后第二批 2/7，
合并 12/18、p=0.238。**单批显著纯是极小可解子集上的噪声。**
规矩因此写死：新开判官臂之前，**先在 10-15 对上只量不一致率、不看胜负**，
并**预注册一条最小可解率**，低于它就不做判官臂。

本脚本就是那一步。它**不回答「3.2 和 4.5 哪个更好看」**，
只回答「判官在这个比较上给不给得出稳定回答」。

## 预注册（写于运行之前，本提交即预注册；事后不改）

**样本框（不由我挑）**：`experiments/prompts_pack78.json` 按**列表原序**，
从头取到攒够 15 对合格对为止。合格 = 两条臂都真的裁了（frac < 0.999）
且两张瓦片不逐像素相同。门（周期 + 各向异性）由 `auto_crop` 内部照常执行。

**两条臂**：同一张 1024 源图，只改 `auto_crop` 的 `target_px`：
  - 真人臂 `target_px = 16 / 3.20`（真人惯例，`resolution_tiers.py`）
  - 设计臂 `target_px = 16 / 4.50`（`UNITS_PER_TILE`，现行管线）
其余一律相同（同源、同降采样 BOX、同 12 色量化、同调色板种子）。
**`UNITS_PER_TILE` 一个字不改**（改它要先有盲比，见 `downsample.py` 注释）。

**两处刻意偏离交付管线，跑前说明理由**：
1. **单样本（best-of 1）**。交付走 best-of 4，但那个选样准则调的是
   `auto_crop(cand, size)` 的**默认 4.5**——用它选出的源图**偏袒设计臂**。
   两条臂必须对称，所以不选样。
2. **居中裁，`seam_align=False`**。接缝对齐是在位置上取极值的搜索，
   两条臂 side 不同、搜索空间也不同，会把尺度变量和位置变量混在一起；
   且所有已发表裁剪数字都出自居中裁。

**问的问题**：必须与将来整轮要问的**完全一致**，否则量到的不一致率不作数——
用本项目一贯那句偏好问题（见 `Q`）。判官 `claude-opus-5`（唯一通过验证的那个）。
正反两种顺序各问一次，温度 0。

**只记可解与否，不记胜负。** 这不是靠自觉：`PAIR_FIELDS` 钉死了可写字段，
写盘前 assert 记录里没有别的键，胜负从来不进内存以外的任何地方。
（温度 0 下判官确定，将来真跑整轮时重问一遍即可，什么都没浪费。）

**空对照（floor）**：另取 5 对，两边**同一张图**（都用设计臂那张）。
判官对同一张图不可能有内容上的偏好，它在这 5 对上的"可解率"就是
纯靠位置偏好/随机断连凑出来的**地板**。这条是判读的一部分：
可解率若不高于地板，那点可解率就不含信息。

**判据表**：设真实对可解率 r = 可解/已答。

| 条件 | 结论 | 退出码 |
| --- | --- | --- |
| 已答不足 12 对（API 失败太多） | 估不出来，本轮不判 | 3 |
| r < 0.60 | **不做判官臂**：改人工盲比，或换个问题 | 2 |
| r >= 0.60 但 r <= 地板 | 同上，**不做判官臂**（可解率不含信息） | 2 |
| r >= 0.60 且 r > 地板 | 判官臂可以做，下一轮渲染整批 | 0 |

**0.60 这条线怎么定的**（跑前定，不是看了数才定）：本项目吃过亏的两个区间是
可辨认性 24% 可解（76% 不一致）和 `spread_rescale` 原轮 48% 可解（弃 11/21）——
**48% 那个区间产出过一个没能复现的假阳性**，所以线必须画在 48% 以上。
0.60 是能同时满足「高于已知会骗人的区间」和「15 对上还测得动」的最低整数刻度。

**n=15 的估计误差要一起报**（Jeffreys 95%），不许拿点估计当准数。
r 落在 0.60 附近而区间跨过它时，结论按点估计走（判据已固定），
但**必须在日志里写明这次判定是压线的**。

**本轮不下任何关于 3.2 与 4.5 谁更好看的结论。** 一个字都不写。

## 两个阶段

    python analysis/paired/units_ab_resolvable.py render   # 要 GPU，在 emnlp 上跑
    python analysis/paired/units_ab_resolvable.py probe    # 要判官，纯网络
    python analysis/paired/units_ab_resolvable.py selftest # 不要 GPU 也不要判官
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

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tools", "analysis", "analysis/annotate"):
    sys.path.insert(0, str(ROOT / sub))
from exact import binom_test, jeffreys                              # noqa: E402

ARM_HUMAN = 3.20          # 真人惯例（resolution_tiers.py，16 与 32 上恒定）
ARM_DESIGN = 4.50         # 现行 UNITS_PER_TILE
SIZE = 16
COLORS = 12
SEED = 21                 # 与 batch_pack 默认一致；单样本即 manual_seed(SEED + mi)
RENDER = 1024
STEPS = 28
N_PAIRS = 15
N_NULL = 5
MIN_RESOLVABLE = 0.60
MIN_ANSWERED = 12
MODEL = "claude-opus-5"

Q = ("下面是两张 {n}x{n} 的像素画材质贴图，材质是「{label}」。\n"
     "哪一张更像这个材质、更像一张能用的游戏贴图？只回答 A 或 B，不要解释。\n"
     "（第一张是 A，第二张是 B）")

# 可写字段白名单。**胜负不在里面，也永远不许加进来**——
# 本脚本的全部意义就是在不看胜负的前提下决定要不要开这条臂。
PAIR_FIELDS = {"pair", "kind", "material", "frac_human", "frac_design",
               "resolved", "answered"}


def b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).resize((256, 256), Image.NEAREST).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def resolved_both_orders(ask, model, q, imgA, imgB, base_url, key):
    """正反各问一次。返回 True(可解)/False(不一致)/None(没答上来)。

    **只返回可解与否**：两次的选择在这个函数里比完就丢掉，不出栈。
    """
    x, y = b64(imgA), b64(imgB)
    o1 = ask(model, q, [x, y], base_url, key)
    o2 = ask(model, q, [y, x], base_url, key)
    if not o1 or not o2:
        return None
    first = "A" if o1.upper().startswith("A") else "B"
    second = "B" if o2.upper().startswith("A") else "A"
    return first == second


def stage_render(a):
    import torch
    from diffusers import StableDiffusionXLPipeline
    from downsample import auto_crop
    from make_texture import extract_palette, quantize
    from from_prompt import TMPL, NEG

    prompts = json.loads((ROOT / "experiments/prompts_pack78.json")
                         .read_text(encoding="utf-8"))
    a.tiles.mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    recs, tried = [], 0
    for mi, m in enumerate(prompts):
        if len(recs) >= N_PAIRS:
            break
        tried += 1
        g = torch.Generator("cuda").manual_seed(SEED + mi)
        im = pipe(TMPL.format(p=m), negative_prompt=NEG, num_inference_steps=STEPS,
                  generator=g, height=RENDER, width=RENDER).images[0]
        src = np.asarray(im).astype(float)
        tiles, fracs = {}, {}
        for tag, upt in (("human", ARM_HUMAN), ("design", ARM_DESIGN)):
            cropped, frac = auto_crop(src, SIZE, target_px=SIZE / upt)
            small = np.asarray(Image.fromarray(cropped.astype(np.uint8))
                               .resize((SIZE,) * 2, Image.BOX))
            # 调色板种子跟着材质走，两条臂用同一个，量化不引入臂间差异
            tiles[tag] = quantize(small, extract_palette(small, COLORS, seed=mi))
            fracs[tag] = frac
        cropped_both = all(f < 0.999 for f in fracs.values())
        same = np.array_equal(tiles["human"], tiles["design"])
        if not cropped_both or same:
            why = "门未触发或钳成整图" if not cropped_both else "两臂瓦片逐像素相同"
            print(f"  [skip] {m:<28} {why}", flush=True)
            continue
        slug = m.replace(" ", "_")
        for tag in ("human", "design"):
            Image.fromarray(tiles[tag].astype(np.uint8)).save(
                a.tiles / f"{slug}_{tag}.png")
        recs.append({"pair": len(recs), "kind": "real", "material": m,
                     "frac_human": fracs["human"], "frac_design": fracs["design"]})
        print(f"  [{len(recs):2d}/{N_PAIRS}] {m:<28} "
              f"frac 3.2={fracs['human']:.3f}  4.5={fracs['design']:.3f}", flush=True)

    # 空对照：前 N_NULL 对的**设计臂**图当两边，跑前定死取哪几对
    for r in recs[:N_NULL]:
        recs.append({"pair": len(recs), "kind": "null", "material": r["material"],
                     "frac_human": r["frac_design"], "frac_design": r["frac_design"]})

    (a.tiles / "pairs.json").write_text(
        json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    n_real = sum(r["kind"] == "real" for r in recs)
    print(f"\n渲染 {tried} 个提示词，得到 {n_real} 对真实对 + {N_NULL} 对空对照")
    if n_real < N_PAIRS:
        print(f"  -> 不足 {N_PAIRS} 对，probe 阶段照跑但要在日志里注明 n")


def stage_probe(a):
    from vlm_judge import ask
    base_url, key = os.environ.get("VLM_BASE_URL"), os.environ.get("VLM_API_KEY")
    if not base_url or not key:
        raise SystemExit("需要环境变量 VLM_BASE_URL 与 VLM_API_KEY")
    recs = json.loads((a.tiles / "pairs.json").read_text(encoding="utf-8"))

    for r in recs:
        slug = r["material"].replace(" ", "_")
        design = np.asarray(Image.open(a.tiles / f"{slug}_design.png").convert("RGB"))
        if r["kind"] == "null":
            left, right = design, design
        else:
            left = np.asarray(Image.open(a.tiles / f"{slug}_human.png").convert("RGB"))
            right = design
        q = Q.format(n=SIZE, label=r["material"])
        v = resolved_both_orders(ask, a.model, q, left, right, base_url, key)
        r["answered"] = v is not None
        r["resolved"] = bool(v) if v is not None else False
        state = ("可解" if v else "不一致") if v is not None else "没答上来"
        print(f"  {r['kind']:<5} {r['material']:<28} {state}", flush=True)

    for r in recs:
        extra = set(r) - PAIR_FIELDS
        assert not extra, f"记录里混进了不该有的字段：{extra}"
    a.out.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    raise SystemExit(verdict(recs))


def verdict(recs) -> int:
    real = [r for r in recs if r["kind"] == "real" and r["answered"]]
    null = [r for r in recs if r["kind"] == "null" and r["answered"]]
    n, k = len(real), sum(r["resolved"] for r in real)
    nn, kn = len(null), sum(r["resolved"] for r in null)
    total_real = sum(r["kind"] == "real" for r in recs)

    print()
    print(f"真实对：已答 {n}/{total_real}，其中可解 {k}")
    if n:
        lo, hi = jeffreys(k, n)
        print(f"  可解率 {k/n:.0%}   Jeffreys 95% [{lo:.0%},{hi:.0%}]"
              f"   不一致率 {(n-k)/n:.0%}")
    floor = kn / nn if nn else float("nan")
    print(f"空对照（同一张图两边）：已答 {nn}/{N_NULL}，可解 {kn}"
          + (f"，地板 {floor:.0%}" if nn else ""))
    print(f"\n预注册最小可解率 {MIN_RESOLVABLE:.0%}；本轮不看胜负，"
          f"记录里也没有胜负（字段白名单 {sorted(PAIR_FIELDS)}）")

    if n < MIN_ANSWERED:
        print(f"判读：已答 {n} 对不足 {MIN_ANSWERED}，可解率估不出来 -> **本轮不判**")
        return 3
    r = k / n
    if r < MIN_RESOLVABLE:
        print(f"判读：可解率 {r:.0%} 低于 {MIN_RESOLVABLE:.0%}")
        print("  -> **不做判官臂**。这个比较判官答不稳，整轮 GPU 会买到一堆弃样，")
        print("     而极小可解子集上的显著已经骗过我一次（2026-09-11 的 10/11）。")
        print("     出路：改人工盲比（v2 两序去偏仪器），或换一个判官答得了的问题。")
        return 2
    if nn and r <= floor:
        print(f"判读：可解率 {r:.0%} 不高于空对照地板 {floor:.0%}")
        print("  -> **不做判官臂**。这点可解率靠位置偏好就能凑出来，不含内容信息。")
        return 2
    lo, hi = jeffreys(k, n)
    print(f"判读：可解率 {r:.0%} 不低于 {MIN_RESOLVABLE:.0%}"
          + (f" 且高于地板 {floor:.0%}" if nn else ""))
    print("  -> 判官臂**可以做**。下一轮才渲染整批、才问胜负。")
    if lo < MIN_RESOLVABLE:
        print(f"     注意：这是**压线判定**——Jeffreys 下界 {lo:.0%} 仍在阈值之下，")
        print("     整轮跑完必须同时报有效 n / 总 n（cron_prompt.md 第 3 条）。")
    return 0


def stage_selftest(a):
    """不碰 GPU、不碰判官，只验判据分支与那条"不许记胜负"的护栏。"""
    def mk(n_real, k_real, n_null, k_null, unanswered=0):
        recs = []
        for i in range(n_real):
            recs.append({"pair": i, "kind": "real", "material": f"m{i}",
                         "frac_human": 0.2, "frac_design": 0.3,
                         "answered": i >= unanswered,
                         "resolved": i >= unanswered and i - unanswered < k_real})
        for j in range(n_null):
            recs.append({"pair": 100 + j, "kind": "null", "material": f"m{j}",
                         "frac_human": 0.3, "frac_design": 0.3,
                         "answered": True, "resolved": j < k_null})
        return recs

    cases = [
        ("可解率高、地板低 -> 开",            mk(15, 12, 5, 1), 0),
        ("可解率恰好压在 0.60 -> 开（压线）",  mk(15, 9, 5, 1), 0),
        ("可解率 53% -> 不开",                mk(15, 8, 5, 1), 2),
        ("可解率 24%（可辨认性那个区间）",     mk(15, 4, 5, 1), 2),
        ("可解率 48%（spread 那个区间）",      mk(15, 7, 5, 1), 2),
        ("可解率 80% 但地板也 80% -> 不开",    mk(15, 12, 5, 4), 2),
        ("API 挂掉一半 -> 不判",              mk(15, 12, 5, 1, unanswered=6), 3),
        ("无空对照也能判",                    mk(15, 12, 0, 0), 0),
    ]
    bad = 0
    for name, recs, want in cases:
        got = verdict(recs)
        flag = "PASS" if got == want else "**FAIL**"
        if got != want:
            bad += 1
        print(f"[{flag}] {name}: 退出码 {got}（期望 {want}）\n" + "-" * 60)

    # 护栏：混进胜负字段必须被 assert 拦住
    leaked = mk(3, 2, 0, 0)
    leaked[0]["winner"] = "human"
    caught = False
    try:
        for r in leaked:
            assert not (set(r) - PAIR_FIELDS)
    except AssertionError:
        caught = True
    print(f"[{'PASS' if caught else '**FAIL**'}] 胜负字段被白名单拦下")
    bad += not caught

    # 护栏：两条臂的 target_px 真的不同，且 3.2 那条裁得更小
    th, td = SIZE / ARM_HUMAN, SIZE / ARM_DESIGN
    ok = th > td
    print(f"[{'PASS' if ok else '**FAIL**'}] target_px 3.2->{th:.3f} 大于 4.5->{td:.3f}"
          f"（同周期下 3.2 裁的窗口更小、单元更大）")
    bad += not ok

    print(f"\n{'全部通过' if not bad else str(bad) + ' 项失败'}")
    raise SystemExit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["render", "probe", "selftest"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tiles", type=Path, default=ROOT / "experiments/units_ab")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "experiments/units_ab_resolvable.json")
    a = ap.parse_args()
    {"render": stage_render, "probe": stage_probe, "selftest": stage_selftest}[a.stage](a)


if __name__ == "__main__":
    main()
