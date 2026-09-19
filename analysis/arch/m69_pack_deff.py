#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(M69) 盲写判读器：配对 CLIP 的逐材质 D 到底按不按**包**聚集？——给候选 `V_mat32` 定价。

⛔ 本文件随预注册（`docs/arch_progress.md` 2026-09-19 16:40 那一节）提交，跑完一个字不许改。
⛔ 写它的时候任何 D 数值都还没被打开：本轮跑前只看过**包计数**（已在预注册正文声明）。

判据全文见预注册第四/五/六节，这里只复述会被代码执行的那几条：

  deff = Var_pack(mean D) / Var_mat(mean D)
      Var_mat  : 对 n 个材质有放回自助 B=20000（seed 6901）
      Var_pack : 按包有放回抽 n_packs 个包、取中选包全部材质，B=20000（seed 6902）

  PACK_DEFF_SMALL : deff <  1.5   （包聚类吃不掉增益 ⇒ V_mat32 的 SE 改善落在 1.26x 那端）
  PACK_DEFF_LARGE : deff >= 1.5   （落在 1.18x 那端；⚠ 只登记，⛔ 本轮不撤销任何已发表判决）

(V0) 分辨力检验（任一不过 ⇒ VOID_RULER_BLIND，后面不判）：
  (V0a) 阴性哨兵：材质随机重分到 3 个**同样大小**的假包，200 次（seed 6903），要求 deff 中位数 < 1.5
  (V0b) 阳性哨兵：真包 1（材质最多者）全体 D 加 +2*sd(D)，要求 deff >= 1.5

(OP1) 料齐 / (OP2) 复现 (M61) mean_D=+0.0336053471228311（tol 5e-4） / (OP3) 包归属自洽。

⚠ 披露（预注册未写死、此处声明的实现细节）：(V0a) 那 200 次重分配里，Var_pack 用 B=2000
  自助（200x20000 太慢），Var_mat 沿用主判据那次 B=20000 的值。⛔ 这不改任何判据阈值。

用法：
    python analysis/arch/m69_pack_deff.py --out experiments/m69_pack_deff.json
    python analysis/arch/m69_pack_deff.py --selftest
"""
import argparse
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from m61_read_noninf16 import (  # noqa: E402  冻结判读器本体，⛔ 不重写
    arm_mean as arm_mean16,
    load_arm as load_arm16,
    mean_sd,
)

DEFF_BREAKEVEN = 1.5              # 盈亏平衡点（预注册第四节）
M61_MEAN_D = 0.0336053471228311   # experiments/m61_noninf16.json: main.mean_D
M61_TOL = 5e-4
N_MAT_16 = 124
N_PACK_16 = 3
MIN_MAT_PER_PACK = 10
B_MAIN = 20000
B_SENTINEL = 2000
N_RESHUFFLE = 200
SEED_MAT = 6901
SEED_PACK = 6902
SEED_SHUFFLE = 6903
K16 = 28


# ---------------------------------------------------------------- 统计

def boot_var_mat(d, b=B_MAIN, seed=SEED_MAT):
    """材质级有放回自助 ⇒ mean D 的方差。"""
    rng = random.Random(seed)
    n = len(d)
    means = []
    for _ in range(b):
        s = 0.0
        for _ in range(n):
            s += d[rng.randrange(n)]
        means.append(s / n)
    return mean_sd(means)[1] ** 2


def boot_var_pack(groups, b=B_MAIN, seed=SEED_PACK):
    """包级有放回自助（抽 n_packs 个包、取其全部材质）⇒ mean D 的方差。"""
    rng = random.Random(seed)
    g = len(groups)
    means = []
    for _ in range(b):
        tot, cnt = 0.0, 0
        for _ in range(g):
            grp = groups[rng.randrange(g)]
            tot += sum(grp)
            cnt += len(grp)
        means.append(tot / cnt)
    return mean_sd(means)[1] ** 2


def deff_of(d, groups, var_mat=None, b_pack=B_MAIN, seed_pack=SEED_PACK):
    vm = boot_var_mat(d) if var_mat is None else var_mat
    vp = boot_var_pack(groups, b=b_pack, seed=seed_pack)
    if vm <= 0:
        return float("inf") if vp > 0 else 1.0
    return vp / vm


def hhi(counts):
    n = float(sum(counts))
    return sum((c / n) ** 2 for c in counts)


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def quantile(xs, q):
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


# ---------------------------------------------------------------- 包归属

def prompt_pack_map(size):
    """提示词 -> 包（名下目标里出现次数最多的那个；平局取迭代顺序里先出现的）。

    与 `colour_task.targets()` 走同一条循环（同一个 load / is_material / by_mat 过滤）。
    """
    sys.path.insert(0, os.path.join(ROOT, "model"))
    sys.path.insert(0, os.path.join(ROOT, "eval"))
    from tiles_data import load
    from colour_task import load_set, is_material

    prompts, split = load_set("V_mat")
    by_mat = {e["material"]: e for e in prompts}
    tally, order = {}, {}
    for s in load(size, split):
        if s["material"] not in by_mat or not is_material(s["material"]):
            continue
        p = by_mat[s["material"]]["prompt"]
        d = tally.setdefault(p, {})
        d[s["pack"]] = d.get(s["pack"], 0) + 1
        order.setdefault((p, s["pack"]), len(order))
    return {p: majority(d, p, order) for p, d in tally.items()}, tally


def majority(counts, prompt, order):
    best = None
    for pack, c in counts.items():
        key = (-c, order[(prompt, pack)])
        if best is None or key < best[0]:
            best = (key, pack)
    return best[1]


def group_by_pack(dmap, pmap):
    g = {}
    for m, v in dmap.items():
        g.setdefault(pmap[m], []).append(v)
    return g


# ---------------------------------------------------------------- 判决

def decide(ops, v0a_median, v0b_deff, deff):
    """⛔ 顺序写死：OP -> (V0) -> 主判据。"""
    for name in ("OP1", "OP2", "OP3"):
        if not ops[name]["ok"]:
            return ops[name]["void"]
    if not (v0a_median < DEFF_BREAKEVEN):
        return "VOID_RULER_BLIND"
    if not (v0b_deff >= DEFF_BREAKEVEN):
        return "VOID_RULER_BLIND"
    return "PACK_DEFF_SMALL" if deff < DEFF_BREAKEVEN else "PACK_DEFF_LARGE"


# ---------------------------------------------------------------- 主流程

def run16(d16):
    ctrl, _, _, cprob = load_arm16(d16, "m61_ctrl", list(range(K16)))
    trt, _, _, tprob = load_arm16(d16, "m61_more", list(range(K16)))
    ops = {}
    mats_c = set.intersection(*[set(v) for v in ctrl.values()]) if ctrl else set()
    mats_t = set.intersection(*[set(v) for v in trt.values()]) if trt else set()
    ok1 = (len(ctrl) == K16 and len(trt) == K16
           and len(mats_c) == N_MAT_16 and mats_c == mats_t)
    ops["OP1"] = {"ok": bool(ok1), "void": "VOID_NO_DATA",
                  "n_ctrl": len(ctrl), "n_trt": len(trt),
                  "n_mat_ctrl": len(mats_c), "n_mat_trt": len(mats_t),
                  "same_materials": mats_c == mats_t,
                  "problems": [cprob, tprob]}
    if not ok1:
        return {"ops": ops, "verdict": ops["OP1"]["void"]}

    mats = sorted(mats_c)
    a, b = arm_mean16(trt, mats), arm_mean16(ctrl, mats)
    dmap = {m: a[m] - b[m] for m in mats}
    d = [dmap[m] for m in mats]
    mean_d, sd_d = mean_sd(d)

    ops["OP2"] = {"ok": abs(mean_d - M61_MEAN_D) < M61_TOL,
                  "void": "VOID_NOT_REPRODUCED",
                  "mean_D": mean_d, "published": M61_MEAN_D}

    pmap, tally = prompt_pack_map(16)
    missing = [m for m in mats if m not in pmap]
    groups = {} if missing else group_by_pack(dmap, pmap)
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    ops["OP3"] = {"ok": (not missing and len(groups) == N_PACK_16
                         and all(s >= MIN_MAT_PER_PACK for s in sizes)),
                  "void": "VOID_PACK_MAP",
                  "n_unmapped": len(missing), "n_packs": len(groups),
                  "pack_sizes": {k: len(v) for k, v in groups.items()}}
    if not ops["OP2"]["ok"] or not ops["OP3"]["ok"]:
        return {"ops": ops, "verdict": decide(ops, 0.0, 9.9, 1.0)}

    glist = [groups[k] for k in sorted(groups)]
    var_mat = boot_var_mat(d)
    var_pack = boot_var_pack(glist)
    deff = var_pack / var_mat

    # (V0a) 阴性哨兵：随机重分到同样大小的假包
    rng = random.Random(SEED_SHUFFLE)
    shuffled = list(d)
    v0a = []
    for i in range(N_RESHUFFLE):
        rng.shuffle(shuffled)
        fake, pos = [], 0
        for s in sizes:
            fake.append(shuffled[pos:pos + s])
            pos += s
        v0a.append(boot_var_pack(fake, b=B_SENTINEL, seed=SEED_PACK + 1 + i) / var_mat)

    # (V0b) 阳性哨兵：最大的真包整体 +2*sd(D)
    big = max(groups, key=lambda k: len(groups[k]))
    shift = 2.0 * sd_d
    dmap_pos = {m: dmap[m] + (shift if pmap[m] == big else 0.0) for m in mats}
    d_pos = [dmap_pos[m] for m in mats]
    g_pos = group_by_pack(dmap_pos, pmap)
    deff_pos = (boot_var_pack([g_pos[k] for k in sorted(g_pos)])
                / boot_var_mat(d_pos))

    verdict = decide(ops, median(v0a), deff_pos, deff)
    return {
        "ops": ops, "verdict": verdict,
        "n_materials": len(mats), "mean_D": mean_d, "sd_D": sd_d,
        "pack_sizes": {k: len(v) for k, v in groups.items()},
        "inv_hhi_packs": 1.0 / hhi(sizes),
        "var_mat": var_mat, "var_pack": var_pack, "deff": deff,
        "se_mat": math.sqrt(var_mat), "se_pack": math.sqrt(var_pack),
        "ci95_mat": [mean_d - 1.96 * math.sqrt(var_mat),
                     mean_d + 1.96 * math.sqrt(var_mat)],
        "ci95_pack": [mean_d - 1.96 * math.sqrt(var_pack),
                      mean_d + 1.96 * math.sqrt(var_pack)],
        "v0a_median": median(v0a), "v0a_p95": quantile(v0a, 0.95),
        "v0a_max": max(v0a), "v0b_deff": deff_pos, "v0b_shift": shift,
        "v0b_pack": big,
    }


def run32(d, ctag, ttag, k, label):
    """只登记：32px 两批数据上的同一个 deff（2 个包、1 自由度 ⇒ ⛔ 只当量级提示）。"""
    from m60_read_scale import arm_mean as arm_mean32, load_arm as load_arm32
    ctrl, _, cprob = load_arm32(d, ctag, k)
    trt, _, tprob = load_arm32(d, ttag, k)
    if cprob or tprob or len(ctrl) != k or len(trt) != k:
        return {"label": label, "ok": False, "problems": [cprob, tprob]}
    mats = sorted(set.intersection(*[set(v) for v in ctrl.values()]))
    a, b = arm_mean32(trt, mats), arm_mean32(ctrl, mats)
    dmap = {m: a[m] - b[m] for m in mats}
    pmap, _ = prompt_pack_map(32)
    if any(m not in pmap for m in mats):
        return {"label": label, "ok": False, "problems": "unmapped materials"}
    groups = group_by_pack(dmap, pmap)
    dd = [dmap[m] for m in mats]
    vm = boot_var_mat(dd)
    vp = boot_var_pack([groups[k2] for k2 in sorted(groups)])
    return {"label": label, "ok": True, "n_materials": len(mats),
            "pack_sizes": {k2: len(v) for k2, v in groups.items()},
            "deff": vp / vm, "se_mat": math.sqrt(vm), "se_pack": math.sqrt(vp),
            "note": "2 packs = 1 df; magnitude hint only, NOT a verdict"}


def vmat32_projection():
    """只登记：`V_mat32` 的包组成与有效 n 增益（解析，跑前已写在预注册第三节）。"""
    now = {"ROllerozxa__mtg_tiled_32x": 66, "hilol__textures__": 8}
    add = {"Winter94__wintercore_dwemer": 22, "ROllerozxa__mtg_tiled_32x": 17}
    new = dict(now)
    for k, v in add.items():
        new[k] = new.get(k, 0) + v
    inv_now, inv_new = 1.0 / hhi(list(now.values())), 1.0 / hhi(list(new.values()))
    return {
        "targets_now": sum(now.values()), "targets_new": sum(new.values()),
        "n_mat_now": 67, "n_mat_new": 106,
        "packs_now": now, "packs_new": new,
        "inv_hhi_now": inv_now, "inv_hhi_new": inv_new,
        "gain_eff_n_by_hhi": inv_new / inv_now,
        "gain_eff_n_rho0": 106.0 / 67.0,
        "gain_eff_n_rho1": 3.0 / 2.0,
        "se_gain_rho0": math.sqrt(106.0 / 67.0),
        "se_gain_rho1": math.sqrt(3.0 / 2.0),
        "se_gain_by_hhi": math.sqrt(inv_new / inv_now),
    }


# ---------------------------------------------------------------- selftest

def selftest():
    ok = 0

    def chk(cond, msg):
        nonlocal ok
        assert cond, msg
        ok += 1

    chk(abs(boot_var_mat([1.0] * 20, b=500)) < 1e-18, "const data -> zero var")
    chk(abs(boot_var_pack([[2.0] * 5, [2.0] * 5, [2.0] * 5], b=500)) < 1e-18,
        "const groups -> zero var")
    chk(abs(hhi([1, 1, 1, 1]) - 0.25) < 1e-12, "hhi equal")
    chk(abs(hhi([10]) - 1.0) < 1e-12, "hhi single")
    chk(median([3.0, 1.0, 2.0]) == 2.0, "median odd")
    chk(median([1.0, 2.0, 3.0, 4.0]) == 2.5, "median even")

    # 平局取先出现的那个包
    order = {("p", "A"): 5, ("p", "B"): 2}
    chk(majority({"A": 3, "B": 3}, "p", order) == "B", "tie -> first seen")
    chk(majority({"A": 4, "B": 9}, "p", order) == "B", "majority wins")

    rng = random.Random(0)
    # 无包结构 ⇒ deff ~ 1
    d = [rng.gauss(0, 1) for _ in range(120)]
    g = [d[0:40], d[40:80], d[80:120]]
    chk(deff_of(d, g, b_pack=4000) < DEFF_BREAKEVEN, "iid data -> deff < 1.5")
    # 强包结构 ⇒ deff 很大
    d2 = [rng.gauss(0, 1) + (6.0 if i < 40 else 0.0) for i in range(120)]
    g2 = [d2[0:40], d2[40:80], d2[80:120]]
    chk(deff_of(d2, g2, b_pack=4000) > 3.0, "clustered data -> deff > 3")

    good = {n: {"ok": True, "void": "V_" + n} for n in ("OP1", "OP2", "OP3")}
    chk(decide(good, 1.0, 5.0, 1.2) == "PACK_DEFF_SMALL", "small")
    chk(decide(good, 1.0, 5.0, 1.5) == "PACK_DEFF_LARGE", "breakeven is LARGE")
    chk(decide(good, 1.0, 5.0, 9.0) == "PACK_DEFF_LARGE", "large")
    chk(decide(good, 1.9, 5.0, 1.2) == "VOID_RULER_BLIND", "V0a fails")
    chk(decide(good, 1.0, 1.1, 1.2) == "VOID_RULER_BLIND", "V0b fails")
    bad = dict(good)
    bad["OP2"] = {"ok": False, "void": "VOID_NOT_REPRODUCED"}
    chk(decide(bad, 1.0, 5.0, 1.2) == "VOID_NOT_REPRODUCED", "OP before V0")
    bad2 = {"OP1": {"ok": False, "void": "VOID_NO_DATA"},
            "OP2": {"ok": False, "void": "VOID_NOT_REPRODUCED"},
            "OP3": {"ok": True, "void": "VOID_PACK_MAP"}}
    chk(decide(bad2, 1.0, 5.0, 1.2) == "VOID_NO_DATA", "OP1 first")

    pr = vmat32_projection()
    chk(abs(pr["gain_eff_n_rho1"] - 1.5) < 1e-12, "rho=1 gain 1.5")
    chk(pr["inv_hhi_new"] > pr["inv_hhi_now"], "hhi improves")
    print("selftest OK %d/%d" % (ok, 19))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir16", default=os.path.join(ROOT, "remote_tmp"))
    ap.add_argument("--dir_m60", default=os.path.join(ROOT, "remote_tmp", "m60_read"))
    ap.add_argument("--dir_m62", default=os.path.join(ROOT, "remote_tmp", "m62"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return

    res = {"main_16px": run16(a.dir16),
           "registered_only_32px": [
               run32(a.dir_m60, "m60_ctrl", "m60_more", 17, "M60 K=17"),
               run32(a.dir_m62, "m62_ctrl", "m62_dbl", 28, "M62 K=28")],
           "vmat32_projection": vmat32_projection()}

    # ⚑ 落盘一律挪到打印之前（崩了数据也在）
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)

    m = res["main_16px"]
    print("=== (M69) 16px 主判据 ===")
    for n in ("OP1", "OP2", "OP3"):
        o = m["ops"].get(n)
        if o:
            print("  %s ok=%s %s" % (n, o["ok"],
                                     {k: v for k, v in o.items()
                                      if k not in ("ok", "void", "problems")}))
    if "deff" in m:
        print("  n=%d  mean_D=%+.6f  sd_D=%.4f  packs=%s  1/HHI=%.2f"
              % (m["n_materials"], m["mean_D"], m["sd_D"], m["pack_sizes"],
                 m["inv_hhi_packs"]))
        print("  SE  material-level %.5f   pack-level %.5f" % (m["se_mat"], m["se_pack"]))
        print("  CI95 material [%+.4f,%+.4f]  pack [%+.4f,%+.4f]"
              % (m["ci95_mat"][0], m["ci95_mat"][1],
                 m["ci95_pack"][0], m["ci95_pack"][1]))
        print("  deff = %.3f   (V0a) null median %.3f p95 %.3f max %.3f"
              % (m["deff"], m["v0a_median"], m["v0a_p95"], m["v0a_max"]))
        print("  (V0b) shifted-pack deff %.3f  (pack=%s shift=%+.4f)"
              % (m["v0b_deff"], m["v0b_pack"], m["v0b_shift"]))
    print("  [判] %s" % m["verdict"])

    print("=== 只登记：32px（2 个包 = 1 自由度，仅量级提示）===")
    for r in res["registered_only_32px"]:
        if r.get("ok"):
            print("  %-10s n=%d packs=%s  deff=%.3f  SE mat %.5f / pack %.5f"
                  % (r["label"], r["n_materials"], r["pack_sizes"], r["deff"],
                     r["se_mat"], r["se_pack"]))
        else:
            print("  %-10s NOT OK %s" % (r["label"], r["problems"]))

    p = res["vmat32_projection"]
    print("=== 只登记：V_mat32 预测 ===")
    print("  包组成 %s -> %s" % (p["packs_now"], p["packs_new"]))
    print("  1/HHI %.2f -> %.2f   有效 n 增益 rho=0 %.2fx / HHI %.2fx / rho=1 %.2fx"
          % (p["inv_hhi_now"], p["inv_hhi_new"], p["gain_eff_n_rho0"],
             p["gain_eff_n_by_hhi"], p["gain_eff_n_rho1"]))
    print("  SE 改善 %.3fx (rho=0) / %.3fx (HHI) / %.3fx (rho=1)"
          % (p["se_gain_rho0"], p["se_gain_by_hhi"], p["se_gain_rho1"]))
    if a.out:
        print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
