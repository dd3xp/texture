#!/usr/bin/env python
"""纯平铺上界：一张**不含任何 32px 内容**的图能把 FD-DINOv2 刷到多少？零 API、近零 GPU。

# 为什么要跑
`scripts/trd_tile16_train_eval.sh` 那轮（`--p_tile16 0.5`）的读数里有一处刺眼的不对称：
**FD-DINOv2 在两个配置上都掉了 ~147/151（35%），而 KID 一个改善一个退步**，同时
`tile2_degeneracy.json` 显示实验组的"四象限每通道绝对差"中位数从 **9.47 塌到 0.69**
（判据③按"逐像素全同 >20%"画的门，实测 0% → 门过了，判决不改）。
两件事很可能是同一件：模型开始输出**近似周期 16** 的 32px 图，而 FD 恰好吃这一套。

在拿这 147 去推动任何架构决策之前，必须先知道**这把尺子是不是被周期性刷了**。

# 仪器
`v11dx_tile2` := 控制组 **自己的 16px 产物** 2×2 平铺成 32px。
它按构造周期恰为 16、**一个 32px 自由度都没有**，是"纯退化上界"。
参照集、方法目录约定、取第 0 张的口径与 `eval/run_eval.py` 逐字相同 → 数字与
`eval_tile16_Vmat_32.json` 里的 `v11dx_direct` / `t16x_direct` 直接可比。

# 判据（跑前写死，与本文件一起提交）
读数 = FD-DINOv2（V_mat、32px）；下限 = m=1 噪声下限 **FD 20**（`eval/noise_floor.py`）。
  - **(P1)** `FD(v11dx_tile2) <= FD(t16x_direct) + 20` → **FD_ARTIFACT**：
    一张纯平铺图就能拿到与实验组同档或更好的 FD → 本轮那 147 的改善
    ⛔ **不许读作"32px 变好"**；FD 在 32px 这一档对周期性敏感。
  - **(P2)** `FD(v11dx_tile2) >= FD(v11dx_direct) - 20` → **FD_CLEAN**：
    纯平铺刷不动 FD → 那 147 不能归因于周期性。
  - 两者之间 → **UNDECIDED**，⛔ 不许挑一边说。
  - ⛔ 无论结果如何：**不改本轮 ①②③ 的判决**（②已判不通过）、不授权任何配置变更、
    不改 `final_test.sh`、不改任何默认值、不碰判官、不开新臂。本检验只判**尺子**，不判假设。
  - 描述性、不入判据：KID / FID / CLIP / 平铺率、`t16x_tile2`。
  - ⚠ 已知局限：32px 参照集有 (M16) 那个包级聚集问题 → **FD 的绝对水平脆**；
    本判据只比同一参照下的方法间高低，且结论形如"退化图也能得高分"，不依赖参照的代表性。

# 操作检验（一）：平铺器
build 阶段对每个平铺目录跑一遍 `tile2_degeneracy.scan`：**exact_rate 必须 = 100%、MAD = 0**，
否则平铺器本身有 bug，判据作废（这是仪器验钥，不是结果）。

# 操作检验（二）：参照集本身是不是周期 16（`refmad`，跑前写死，与本段一起提交）
若判出 **FD_ARTIFACT**，"尺子被周期性刷了"这个读法还有一个**竞争解释**：
**真人 32px 参照集本身大多就是"16px 内容平铺两遍"** —— 那样的话平铺图并非"零信息"，
它只是**忠实于数据**，尺子一点毛病没有，而"32px 缺口"要整个重新定义。
两者必须分开，否则 FD_ARTIFACT 不许下。

读数 = 真人 32px 参照瓦片（与 `run_eval.py` 同一批：`load(32, val)` 过 `is_material`）的
**四象限每通道绝对差中位数**。两个阈值都取本轮已发表的读数，跑前固定：
  - 中位 <= **0.69**（= 实验组 `t16x_direct` 的中位）→ **REF_PERIODIC**：参照集自己就是周期 16
    → ⛔ **不许下 FD_ARTIFACT**，只能记"参照集是周期 16"，且该发现须另行预注册再测。
  - 中位 >= **9.47**（= 控制组 `v11dx_direct` 的中位）→ **REF_RICH**：参照集不是周期 16
    → 平铺图确实不含与参照匹配的 32px 结构，**FD_ARTIFACT 的读法成立**。
  - 之间 → **UNDECIDED**，⛔ 不许挑一边说。
稳健性（预注册，遵 (M16)「真人语料以包为单位」）：逐包再算一次中位数；
**若各包不全落在判决阈值的同一侧**，判决加后缀 `_FRAGILE`，⛔ 只能引方向、不许引数值。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tile2_degeneracy import scan                        # noqa: E402

FLOOR_FD = 20.0          # eval/noise_floor.py，m=1


def build(args):
    for tag in args.src:
        src = args.src_root / tag / str(args.size)
        dst = args.out_root / f"{tag}_tile2" / str(args.size * 2)
        if not src.is_dir():
            raise SystemExit(f"无源目录：{src}")
        dst.mkdir(parents=True, exist_ok=True)
        files = sorted(src.glob("*.png"))
        for f in files:
            a = np.asarray(Image.open(f).convert("RGB"))
            Image.fromarray(np.tile(a, (2, 2, 1))).save(dst / f.name)
        r = scan(dst)                                    # 操作检验：必须是 100% / 0.0
        print(f"{tag}_tile2  写出 {len(files)} 张 -> {dst}\n"
              f"  操作检验：四象限全同 {r['exact_rate']:.1%}  MAD 均值 {r['mad_mean']:.4f}", flush=True)
        if r["exact_rate"] != 1.0 or r["mad_mean"] != 0.0:
            raise SystemExit("操作检验不通过：平铺器有 bug，判据作废")


def verdict(args):
    tiled = json.loads(args.tiled.read_text(encoding="utf-8"))
    base = json.loads(args.base.read_text(encoding="utf-8"))
    fd_up = tiled["v11dx_tile2"]["FD_DINOv2"]
    fd_exp = base["t16x_direct"]["FD_DINOv2"]
    fd_ctl = base["v11dx_direct"]["FD_DINOv2"]

    if fd_up <= fd_exp + FLOOR_FD:
        v = "FD_ARTIFACT"
    elif fd_up >= fd_ctl - FLOOR_FD:
        v = "FD_CLEAN"
    else:
        v = "UNDECIDED"

    out = {"FD_v11dx_tile2": fd_up, "FD_t16x_direct": fd_exp, "FD_v11dx_direct": fd_ctl,
           "P1_threshold": fd_exp + FLOOR_FD, "P2_threshold": fd_ctl - FLOOR_FD,
           "floor_FD": FLOOR_FD, "verdict": v,
           "desc": {k: {m: tiled[k].get(m) for m in ("KID_x1e3", "FID", "CLIP", "tile_seam_ratio")}
                    for k in tiled}}
    print(f"\n纯平铺上界 FD = {fd_up:.2f}")
    print(f"  P1 门（<= t16x_direct {fd_exp:.2f} + {FLOOR_FD:.0f} = {fd_exp + FLOOR_FD:.2f}）")
    print(f"  P2 门（>= v11dx_direct {fd_ctl:.2f} - {FLOOR_FD:.0f} = {fd_ctl - FLOOR_FD:.2f}）")
    print(f"判决：{v}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)
        print("->", args.out)


MED_EXP, MED_CTL = 0.69, 9.47        # t16x_direct / v11dx_direct 的四象限 MAD 中位（本轮已发表）


def quad_mad(rgb):
    a = rgb.astype(np.int16)
    h = a.shape[0] // 2
    return float(np.abs(a - np.tile(a[:h, :h], (2, 2, 1))).mean())


def refmad(args):
    sys.path.insert(0, str(ROOT / "model"))
    sys.path.insert(0, str(ROOT / "eval"))
    from tiles_data import load                          # noqa: E402
    from prompts import is_material                      # noqa: E402

    tiles = [s for s in load(args.size, args.split) if is_material(s["material"])]
    by_pack = {}
    mads = []
    for s in tiles:
        m = quad_mad(s["palette"][s["idx"]])
        mads.append(m)
        by_pack.setdefault(s["pack"], []).append(m)
    med = float(np.median(mads))

    if med <= MED_EXP:
        v, thr = "REF_PERIODIC", MED_EXP
    elif med >= MED_CTL:
        v, thr = "REF_RICH", MED_CTL
    else:
        v, thr = "UNDECIDED", None

    packs = {p: float(np.median(x)) for p, x in sorted(by_pack.items())}
    if thr is not None:
        same = all((pm <= thr) == (med <= thr) for pm in packs.values())
        if not same:
            v += "_FRAGILE"

    print(f"真人 {args.size}px 参照（{args.split}）：n={len(tiles)}，{len(packs)} 个包")
    print(f"  四象限每通道绝对差：中位 {med:.3f}  均值 {np.mean(mads):.3f}")
    print(f"  门：REF_PERIODIC <= {MED_EXP}   REF_RICH >= {MED_CTL}")
    for p, pm in packs.items():
        print(f"    {p:44s} n={len(by_pack[p]):3d}  中位 {pm:.3f}")
    print(f"判决：{v}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"n": len(tiles), "median": med, "mean": float(np.mean(mads)),
                   "packs": packs, "pack_n": {p: len(x) for p, x in by_pack.items()},
                   "MED_EXP": MED_EXP, "MED_CTL": MED_CTL, "verdict": v},
                  open(args.out, "w"), indent=1)
        print("->", args.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--src", nargs="+", default=["v11dx", "t16x"])
    b.add_argument("--src_root", type=Path, default=ROOT / "experiments/baselines")
    b.add_argument("--out_root", type=Path, default=Path("/tmp/tile2_up"))
    b.add_argument("--size", type=int, default=16)
    b.set_defaults(fn=build)
    j = sub.add_parser("verdict")
    j.add_argument("--tiled", type=Path, required=True)
    j.add_argument("--base", type=Path, required=True)
    j.add_argument("--out", type=Path, default=None)
    j.set_defaults(fn=verdict)
    r = sub.add_parser("refmad")
    r.add_argument("--size", type=int, default=32)
    r.add_argument("--split", default="val")
    r.add_argument("--out", type=Path, default=None)
    r.set_defaults(fn=refmad)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
