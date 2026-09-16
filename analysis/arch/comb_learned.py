#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(M29) 判决脚本：把表按画布解绑重训之后，模型有没有**学到**真人 32px 那支被漏掉的成分？

预注册全文 = `scripts/trd_sizecond_train_eval.sh` 的头部 + `docs/arch_progress.md` 的 (M29) 条目，
判据随那个脚本一起提交、**先于任何数据**。本文件只实现那些判据，⛔ 一个字不许放宽。
零判官 / 零 API / 零 GPU：只读两臂已生成的 PNG。

## 问的是什么

(M25) 量到真人 32px 除"随画布缩放"的主周期外**另有一支固定 4 像素的成分**
（C(4)=+0.010、C(12)=+0.011，包级 CI [+0.0066,+0.0178]），而 v10/v11d 的产物**没继承**
（(M28) 控制臂 C(4)=-0.0020、C(12)=+0.0002）。(M26) 给了构造性的理由：表只吃 u=d/n，
而**同一个 u 上两档要求相反的符号** -> 一张表装不下两档。(M27)/(M28) 证明那张表**确实是把手**。

于是 (M29)：把表按画布解绑（零初始化 FiLM，输入只有 s=n/32）重训，看新臂的 32px 产物
会不会长出 d=4 这支梳齿。**控制臂 = v11d**（(M13) 的同一个控制组，配方逐字相同，唯一变量是
`--bias_size_cond`）。

⚠ **与 (M28) 的区别**：(M28) 是**推理期**干预，干预臂分布外（A_hat(1) 0.166->0.083），
只能说"表**能**支配周期"。本轮两臂都是**正常训练出来的模型**，所以问的是"解绑之后模型
**学不学得到**那支成分"——这才是能决定架构去留的那一问。

## 判据（跑前写死）

- **② 主判据**：ΔC(4)=C_new(4)-C_ctl(4) 族级 CI 下界 > 0 **且** C_new(4) 族级 CI 下界 > 0。
  合取保护直接引 (M27)/(M28) 实测的空总体标定：纯衰减总体上 C(4) 恒为负 -> 正的绝对 C(4)
  凸性造不出来。⚠ **只用 d=4 一个位置**（不做多重比较）；C(12) 只作 (R3) 描述性。
- **③ 操作检验（任一条不过 -> `OPS_FAILED`，不下判）**：
  (OP1) 新臂 A_hat(1) >= 控制臂的 0.8 倍；(OP2) flat_frac <= 控制臂 + 0.05；
  (OP3) >= 500 对且 >= 150 族。
- **⑤ 判决**：②③全过 -> `UNBIND_WORKS`；ΔC(4) 族级 CI **上界 < HUMAN_LO=+0.0066**
  （真人那支效应量的包级 CI 下界）-> `UNBIND_NULL`；其余 `UNDECIDED`。
  ⛔ `UNBIND_NULL` 只能读成"解绑到位了也补不出那一支"，⛔ 不许读成"表跟结构无关"
  （(M27)/(M28) 已经反证）。⛔ `UNDECIDED` 不许往任何一边读。
- **① 守门**：16px KID <= 7.2 由 `--eval16` 指向的 JSON 读；不过 -> `GATE_FAILED`，②③**不下判**。

## 识别检验（⑥；必须先跑，不过就不看真数据）

- **(ID1)** ctl=纯衰减、new=周期 4 -> 必须判 `UNBIND_WORKS`（长出来了看得见）。
- **(ID2)** 两臂都是周期 8 -> 必须判 `UNBIND_NULL`（尺子报得出"什么也没长出来"）。
- **(ID3)** new 臂退化成近纯色 -> 必须判 `OPS_FAILED`（退化门真的会响，不会把退化读成"学到了"）。
- **(ID4) 阳性对照（本轮特有）**：同一条代码路径在 (M28) 已存的 `m28_ctl`/`m28_over` 上
  必须判 `UNBIND_WORKS` —— 那是这把尺子唯一一次在**真产物**上量到过"梳齿长出来"。
  ⚠ 这不是本轮的数据，是 2026-09-16 (M28) 已发表的产物，看它不构成偷看。

用法：
    python analysis/arch/comb_learned.py --gen_root /tmp/gen32_m29 \
        --ctl_tag m29_ctl --new_tag m29_new --id4_root /tmp/gen32_m28 \
        --eval16 /tmp/runs/trd_scond_09161230/eval_scond_Vmat_16.json --out /tmp/comb_learned.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from period_scale import contrast                  # noqa: E402  局部对比 C(d)，逐字复用
from scale_prior import tile_curves                # noqa: E402  A_hat 曲线，逐字复用
# 下面四个连同合成总体一起从 (M28) 逐字复用：同一把尺子，换的只是判决函数
from comb_shift import (N, B_BOOT, SEED, family_of, flat_frac, load_arm,  # noqa: E402
                        ci, checker, decay)

D_MAIN = 4              # 真人 32px 那支固定 4px 成分的位置（(M25)）
HUMAN_LO = 0.0066       # (M25) 真人那支效应量的包级 CI 下界 —— `UNBIND_NULL` 的等价界，⛔ 不许改
KID_GATE = 7.2          # 判据①：v11d 的 6.2 + 1.0，与 (M13) ① 逐字相同
MIN_PAIRS = 500
MIN_FAMS = 150
A1_RATIO = 0.8          # (OP1)：新臂近邻相关不许掉到控制臂的 0.8 倍以下
FLAT_SLACK = 0.05       # (OP2)


def decide(d4_ci, c4_new_ci, ops_ok):
    """跑前写死的判决函数；ID1/ID2/ID3/ID4 走的是同一个函数。"""
    if not ops_ok:
        return "OPS_FAILED"
    if d4_ci[0] > 0 and c4_new_ci[0] > 0:
        return "UNBIND_WORKS"
    if d4_ci[1] < HUMAN_LO:
        return "UNBIND_NULL"
    return "UNDECIDED"


def boot(cc, cn, fams, b=B_BOOT, seed=SEED, paired=True):
    """族级自助 -> (ΔC(4), C_new(4)) 各 b 个值。paired=False 时两臂独立重采样（R2）。"""
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for i, f in enumerate(fams):
        by[f].append(i)
    groups = [np.asarray(by[k]) for k in sorted(by)]
    d4, n4 = [], []
    for _ in range(b):
        sel = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        sel2 = sel if paired else np.concatenate(
            [groups[j] for j in rng.integers(0, len(groups), len(groups))])
        mc, mn = cc[sel].mean(0), cn[sel2].mean(0)
        d4.append(contrast(mn, D_MAIN, N) - contrast(mc, D_MAIN, N))
        n4.append(contrast(mn, D_MAIN, N))
    return np.asarray(d4), np.asarray(n4)


def ops(cc, cn, fams, ff_c, ff_n):
    a1c, a1n = float(cc.mean(0)[0]), float(cn.mean(0)[0])
    rec = {"OP1_a_hat_1": [a1c, a1n], "OP1": bool(a1n >= A1_RATIO * a1c),
           "OP2_flat": [ff_c, ff_n], "OP2": bool(ff_n <= ff_c + FLAT_SLACK),
           "OP3_pairs": int(len(cc)), "OP3_families": len(set(fams)),
           "OP3": bool(len(cc) >= MIN_PAIRS and len(set(fams)) >= MIN_FAMS)}
    rec["all"] = bool(rec["OP1"] and rec["OP2"] and rec["OP3"])
    return rec


def report(name, cc, cn, fams, ff_c, ff_n, out, key, paired=True):
    pc, pn = cc.mean(0), cn.mean(0)
    d4, n4 = boot(cc, cn, fams, paired=paired)
    op = ops(cc, cn, fams, ff_c, ff_n)
    rec = {"n_pairs": int(len(cc)), "n_families": len(set(fams)),
           "dC4": float(contrast(pn, D_MAIN, N) - contrast(pc, D_MAIN, N)), "dC4_ci": ci(d4),
           "C4_ctl": float(contrast(pc, D_MAIN, N)), "C4_new": float(contrast(pn, D_MAIN, N)),
           "C4_new_ci": ci(n4),
           "C12_ctl": float(contrast(pc, 12, N)), "C12_new": float(contrast(pn, 12, N)),
           "C8_ctl": float(contrast(pc, 8, N)), "C8_new": float(contrast(pn, 8, N)),
           "curve_ctl": pc.tolist(), "curve_new": pn.tolist(), "op": op,
           "verdict": decide(ci(d4), ci(n4), op["all"])}
    out[key] = rec
    print("[%s] pairs=%d families=%d A_hat(1) ctl=%.4f new=%.4f"
          % (name, len(cc), rec["n_families"], pc[0], pn[0]))
    print("  C(4)  ctl=%+.4f new=%+.4f  dC4=%+.4f family CI [%+.4f, %+.4f]"
          % (rec["C4_ctl"], rec["C4_new"], rec["dC4"], *rec["dC4_ci"]))
    print("  C_new(4) CI [%+.4f, %+.4f]   (R3) C(12) ctl=%+.4f new=%+.4f   C(8) ctl=%+.4f new=%+.4f"
          % (*rec["C4_new_ci"], rec["C12_ctl"], rec["C12_new"], rec["C8_ctl"], rec["C8_new"]))
    print("  [OP] OP1=%s(%.4f vs %.4f) OP2=%s(flat %.3f vs %.3f) OP3=%s"
          % (op["OP1"], op["OP1_a_hat_1"][1], A1_RATIO * op["OP1_a_hat_1"][0],
             op["OP2"], ff_c, ff_n, op["OP3"]))
    print("  -> %s" % rec["verdict"])
    return rec


def decay_comb(n_tiles=600, size=N, r=2, k=4, amp=0.0, seed=11):
    """(ID1/ID2) 逼真的合成总体：`comb_shift.decay` 的平滑高斯场 + 一支**弱的**周期-4 调制。
    ⚑ 为什么不直接用 `checker(2)`（周期 4 的块棋盘）：那种总体的**近邻**相关是负的
    （A_hat(1)≈0），(OP1) 退化门会当场对它开火 —— 于是 (ID1) 测的就变成了门而不是统计量。
    真人 32px 那支成分本来就比主成分小 5-8 倍（(M25)），叠加式的弱调制才是它的样子。"""
    rng = np.random.default_rng(seed)
    ys, xs = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    comb = np.cos(2 * np.pi * xs / D_MAIN) + np.cos(2 * np.pi * ys / D_MAIN)
    rows = []
    for _ in range(n_tiles):
        z = rng.standard_normal((size, size))
        zz = np.zeros_like(z)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                zz += np.roll(np.roll(z, dy, 0), dx, 1)
        zz = zz / zz.std() + amp * comb
        g = np.digitize(zz, np.quantile(zz, np.arange(1, k) / k)).astype(np.int64)
        rows.append({"idx": g, "k_used": int(g.max()) + 1})
    return tile_curves(rows, size)


def solid(n_tiles=600, size=N, seed=13):
    """(ID3) 退化总体：几乎纯色（只有零星噪点）——退化门必须对它响。"""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_tiles):
        g = np.zeros((size, size), dtype=np.int64)
        g[rng.random(g.shape) < 0.01] = 1
        rows.append({"idx": g, "k_used": int(g.max()) + 1})
    return rows


def pair_arms(root: Path, ctl_tag: str, new_tag: str):
    fc, rc = load_arm(root, ctl_tag)
    fn, rn = load_arm(root, new_tag)
    assert [f.name for f in fc] == [f.name for f in fn], "两臂文件名必须逐一对齐"
    keep = [i for i in range(len(rc)) if rc[i]["k_used"] >= 3 and rn[i]["k_used"] >= 3]
    cc = tile_curves([rc[i] for i in keep], N)
    cn = tile_curves([rn[i] for i in keep], N)
    fams = [family_of(fc[i]) for i in keep]
    return cc, cn, fams, flat_frac([rc[i] for i in keep]), flat_frac([rn[i] for i in keep])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_root", default="/tmp/gen32_m29")
    ap.add_argument("--ctl_tag", default="m29_ctl")
    ap.add_argument("--new_tag", default="m29_new")
    ap.add_argument("--id4_root", default="/tmp/gen32_m28",
                    help="(ID4) 阳性对照：(M28) 已发表的两臂（不是本轮数据）")
    ap.add_argument("--eval16", default="", help="判据①的 16px 评测 JSON；不给则只报 ②③")
    ap.add_argument("--out", default="/tmp/comb_learned.json")
    a = ap.parse_args()
    out = {"n": N, "d_main": D_MAIN, "human_lo": HUMAN_LO, "kid_gate": KID_GATE,
           "B": B_BOOT, "seed": SEED, "tags": [a.ctl_tag, a.new_tag]}

    # ---------- ⑥ 识别检验（先跑：不过就不必看真数据） ----------
    ident = {}
    cases = [("ID1_grew", decay_comb(amp=0.0), decay_comb(amp=0.4, seed=12), None, "UNBIND_WORKS"),
             ("ID2_nothing", checker(4, n_tiles=600), checker(4, n_tiles=600, seed=12),
              None, "UNBIND_NULL"),
             ("ID3_degenerate", checker(4, n_tiles=600), tile_curves(solid(), N),
              solid(), "OPS_FAILED")]
    for name, cc, cn, deg_rows, need in cases:
        fams = ["f%d" % (i % 160) for i in range(len(cc))]
        ff_n = flat_frac(deg_rows) if deg_rows is not None else 0.0
        rec = report("ID %s" % name, cc, cn, fams, 0.0, ff_n, ident, name)
        rec["required"] = need
        rec["pass"] = rec["verdict"] == need
        print("  -> 要求 %s : %s" % (need, "PASS" if rec["pass"] else "FAIL"))

    id4_root = Path(a.id4_root)
    if (id4_root / "m28_ctl" / str(N)).exists():
        cc, cn, fams, ffc, ffn = pair_arms(id4_root, "m28_ctl", "m28_over")
        # ⚠ (ID4) 只检验**统计量**，不检验退化门：(M28) 的干预臂按它自己的记载就是**分布外**的
        # （A_hat(1) 0.166->0.083），(OP1) 对它开火是**正确**的。退化门的正例是 (ID3)。
        # 故此处强制 ops_ok=True，只问"结构判决对不对"。⛔ 这不是放宽判据：主判据的 ②③ 一个字没动。
        rec = report("ID4_m28_positive_control", cc, cn, fams, ffc, 0.0, ident, "ID4_m28")
        rec["verdict"] = decide(rec["dC4_ci"], rec["C4_new_ci"], True)
        rec["note"] = "只检验统计量；(M28) 干预臂分布外，退化门对它开火是正确的"
        rec["required"] = "UNBIND_WORKS"
        rec["pass"] = rec["verdict"] == "UNBIND_WORKS"
        print("  (ID4 只判统计量，ops 强制通过) -> %s" % rec["verdict"])
        print("  -> 要求 UNBIND_WORKS : %s" % ("PASS" if rec["pass"] else "FAIL"))
    else:
        ident["ID4_m28"] = {"pass": False, "note": "(M28) 产物不在 %s" % id4_root}
        print("[ID4] 跳过：%s 不存在 -> 识别检验不完整" % id4_root)

    out["identification"] = ident
    id_ok = all(ident[k].get("pass") for k in ident)
    print("\n=== 识别检验 %s ===\n" % ("全过" if id_ok else "**不过**"))

    # ---------- ① 守门 ----------
    gate = {"kid": None, "pass": None}
    if a.eval16:
        ev = json.load(open(a.eval16, encoding="utf-8"))
        kid = None
        for k, v in (ev.get("methods") or ev).items():
            # run_eval.py 写的键是 "KID_x1e3"（"KID" 从来不存在）-> 原来的读法恒为 None，
            # 会把"确实不过门"误报成"读不到"。两个键都试，缺则保持 None。
            if isinstance(v, dict) and "scond" in str(k):
                for key in ("KID_x1e3", "KID"):
                    if v.get(key) is not None:
                        kid = float(v[key])
                        break
        gate = {"kid": kid, "pass": (kid is not None and kid <= KID_GATE)}
        print("[①守门] 16px KID=%s 门槛 %.1f -> %s" % (kid, KID_GATE, gate["pass"]))
    out["gate16"] = gate

    # ---------- ②③ 主判据 ----------
    root = Path(a.gen_root)
    if not (root / a.ctl_tag / str(N)).exists():
        out["verdict"] = "PENDING_NO_DATA"
        print("[主] %s 还没有产物 -> 只跑了识别检验" % root)
        json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return
    cc, cn, fams, ffc, ffn = pair_arms(root, a.ctl_tag, a.new_tag)
    main_rec = report("main %s vs %s @32" % (a.ctl_tag, a.new_tag), cc, cn, fams, ffc, ffn, out, "main")

    verdict = main_rec["verdict"]
    if not id_ok:
        verdict = "VOID_NO_IDENTIFICATION"
    elif gate["pass"] is False:
        verdict = "GATE_FAILED"          # ①不过 -> ②③不下判（跑前写死）

    # ---------- (R1) LOFO / (R2) 非配对 ----------
    if verdict in ("UNBIND_WORKS", "UNBIND_NULL"):
        by = defaultdict(list)
        for i, f in enumerate(fams):
            by[f].append(i)
        flips = []
        for k in sorted(by):
            drop = set(by[k])
            sel = np.asarray([i for i in range(len(cc)) if i not in drop])
            d4, n4 = boot(cc[sel], cn[sel], [fams[i] for i in sel], b=400, seed=1)
            if decide(ci(d4), ci(n4), True) != verdict:
                flips.append(k)
        out["R1_lofo"] = {"families": len(by), "flips": len(flips), "which": flips[:5]}
        print("[R1] LOFO %d 族，翻侧 %d" % (len(by), len(flips)))
        d4u, n4u = boot(cc, cn, fams, paired=False)
        out["R2_unpaired"] = {"dC4_ci": ci(d4u), "verdict": decide(ci(d4u), ci(n4u), True)}
        print("[R2] 非配对 dC4 CI [%+.4f, %+.4f] -> %s"
              % (*out["R2_unpaired"]["dC4_ci"], out["R2_unpaired"]["verdict"]))
        if flips:
            verdict += "_FRAGILE"
        if out["R2_unpaired"]["verdict"] != main_rec["verdict"]:
            verdict += "_PAIRING_SENSITIVE"

    out["verdict"] = verdict
    print("\n==== (M29) 判决：%s ====" % verdict)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("写出 %s" % a.out)


if __name__ == "__main__":
    main()
