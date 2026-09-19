#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(P8) 判读器：环面相对注意力偏置到底承不承重？

判据**逐字照抄**已提交的预注册 `scripts/p8_bias_ablation.sh` 的判据节（commit 7a4c90f），
**盲写**：写这个文件时四条臂一张图都还没生成（C 臂训练中、A1/A2/A3 未开跑），
没有看过任何接缝读数、任何 KID、任何 config.json。已披露的偏倚：无。

零 GPU、零 API、零判官、零活件改动（`tile_seam_ratio` / `load_method` 逐字从 AST 借出来跑，
⛔ 没重写、⛔ 没 import torch）。只读：
  - `<dir>/p8{C,A1,A2,A3}x/16/*.png`          四臂 16px 产物（scp 自远程 experiments/baselines/）
  - `<runs>/trd_p8{C,A1,A2,A3}_<stamp>/`      四臂的 config.json / log.json（(OP1)(OP2) 用）
  - `<dir>/p8_eval_Vmat_16.json`              run_eval 产物（判据②「只登记」用）

用法：
    python analysis/arch/p8_read_bias.py --selftest
    python analysis/arch/p8_read_bias.py --dir remote_tmp/p8 --dry     # 只查料，不判
    python analysis/arch/p8_read_bias.py --dir remote_tmp/p8 --out experiments/p8_bias.json

⚠ 判据里没写死的两处实现选择，**在看到任何读数之前**在此冻结：
  (a) 每个材质有 `--n 2` 张图 ⇒ 逐材质先取**该材质全部有限张的中位数**（n=2 即两张的均值），
      得到该臂该材质的一个 |ratio−1|；瓦片内部近乎平涂时 `tile_seam_ratio` 返回 nan ⇒ 该张丢弃，
      一个材质全部张都是 nan ⇒ 该材质在该臂缺席。
  (b) 主判据的两个中位数与符号检验**都在「两臂都有读数」的配对材质集上算**（口径一致），
      各臂自己的无配对中位数只登记。符号检验用**双侧**精确二项（math.comb，环境无 scipy），
      差值恰为 0 的材质按惯例弃用；单侧 p 只登记。
"""
import argparse
import ast
import json
import math
import os
import re
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STAMP = "09190900"
CTRL = "C"
ABL = ["A1", "A2", "A3"]
ARMS = [CTRL] + ABL
MARGIN = 0.05            # 判据①：中位数要高出控制臂这么多
ALPHA = 0.05             # 判据①：配对符号检验门槛
KID_THR = 4.76           # 判据②：max(重训漂移 1.087, 采样下限 4.76)
N_MAT = 125              # (OP4)：V_mat 条目数
STEPS = 12000            # (OP2)：跑满这么多步
# (OP1)：四臂 config.json 只允许这两个键不同
ALLOWED_DIFF = {"bias_wrap", "bias_off", "out"}


# ---------------------------------------------------------------- 逐字复用活件
def borrow(path, names, ns):
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    want = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names}
    missing = [n for n in names if n not in want]
    if missing:
        raise RuntimeError("source changed, missing %s in %s" % (missing, path))
    mod = ast.Module(body=[want[n] for n in names], type_ignores=[])
    exec(compile(mod, path, "exec"), ns)          # noqa: S102  只跑我们自己的仓库源码
    return ns


def activepieces():
    from pathlib import Path
    ns = {"np": np, "re": re, "Image": Image, "Path": Path}
    borrow(os.path.join(ROOT, "eval", "metrics.py"), ["tile_seam_ratio"], ns)
    borrow(os.path.join(ROOT, "eval", "run_eval.py"), ["load_method"], ns)
    return ns


# ---------------------------------------------------------------- 小工具
def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    return float(s[n // 2]) if n % 2 else float((s[n // 2 - 1] + s[n // 2]) / 2.0)


def finite(xs):
    return [x for x in xs if isinstance(x, float) and math.isfinite(x)]


def binom_two_sided(k, n):
    """精确二项检验 p=0.5 的双侧 p（环境无 scipy ⇒ math.comb）。"""
    if n == 0:
        return float("nan")
    pmf = [math.comb(n, i) for i in range(n + 1)]
    tot = float(sum(pmf))
    obs = pmf[k]
    return float(sum(v for v in pmf if v <= obs + 1e-9) / tot)


def binom_one_sided(k, n):
    """P(X >= k)，用于「消融臂更差」这个方向的单侧 p（只登记）。"""
    if n == 0:
        return float("nan")
    tot = float(2 ** n)
    return float(sum(math.comb(n, i) for i in range(k, n + 1)) / tot)


# ---------------------------------------------------------------- 读料
def slugs_of(d):
    """目录里出现的材质名（`<slug>.png` 或 `<slug>_<k>.png`）。"""
    out = set()
    if not os.path.isdir(d):
        return out
    for fn in os.listdir(d):
        if not fn.endswith(".png"):
            continue
        stem = fn[:-4]
        m = re.fullmatch(r"(.+)_(\d+)", stem)
        out.add(m.group(1) if m else stem)
    return out


def arm_dir(base, arm):
    return os.path.join(base, "p8%sx" % arm, "16")


def per_material_dev(base, arm, slugs, ns):
    """返回 {材质: |ratio−1|}，外加诊断计数。逐张 ratio 用活件 `tile_seam_ratio`。"""
    from pathlib import Path
    d = arm_dir(base, arm)
    _, groups = ns["load_method"](Path(d), slugs)
    dev, n_tile, n_nan, n_missing = {}, 0, 0, 0
    for s, imgs in zip(slugs, groups):
        if not imgs:
            n_missing += 1
            continue
        rs = []
        for im in imgs:
            n_tile += 1
            r = ns["tile_seam_ratio"](np.asarray(im, np.float64).mean(axis=2))
            if math.isfinite(r):
                rs.append(abs(r - 1.0))
            else:
                n_nan += 1
        if rs:
            dev[s] = median(rs)
    return dev, {"tiles": n_tile, "nan_tiles": n_nan, "missing_materials": n_missing,
                 "materials_with_reading": len(dev)}


def load_json(p):
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:       # ⚠ Windows 默认 GBK 会崩在中文字段
        return json.load(f)


# ---------------------------------------------------------------- 操作检验
def op_checks(base, runs):
    """(OP1)(OP2)(OP4) —— (OP3) 生成口径由本脚本对着两个 .sh 的 gen 行逐字核（见 gen_line_ok）。"""
    res = {}
    cfgs = {}
    for a in ARMS:
        cfgs[a] = load_json(os.path.join(runs, "trd_p8%s_%s" % (a, STAMP), "config.json"))
    res["OP1"] = config_diff(cfgs)
    res["OP2"] = {}
    for a in ARMS:
        rd = os.path.join(runs, "trd_p8%s_%s" % (a, STAMP))
        log = load_json(os.path.join(rd, "log.json"))
        last_step = max([r.get("step", 0) for r in log], default=0) if isinstance(log, list) else 0
        res["OP2"][a] = {"step_ckpt": os.path.exists(os.path.join(rd, "step_%d.pt" % STEPS)),
                         "last_pt": os.path.exists(os.path.join(rd, "last.pt")),
                         "last_step": last_step,
                         "ok": last_step >= STEPS}
    res["OP4"] = {}
    for a in ARMS:
        s = slugs_of(arm_dir(base, a))
        res["OP4"][a] = {"materials": len(s), "ok": len(s) == N_MAT}
    ref = slugs_of(arm_dir(base, CTRL))
    res["OP4"]["same_slugs"] = all(slugs_of(arm_dir(base, a)) == ref for a in ABL) and bool(ref)
    return res


def config_diff(cfgs):
    have = [a for a in ARMS if cfgs.get(a)]
    if len(have) < len(ARMS):
        return {"ok": False, "reason": "missing_config", "have": have}
    keys = set()
    for a in have:
        keys |= set(cfgs[a].keys())
    bad = {}
    for k in sorted(keys):
        vals = {a: cfgs[a].get(k, "<absent>") for a in have}
        if len(set(json.dumps(v, sort_keys=True, default=str) for v in vals.values())) > 1:
            if k not in ALLOWED_DIFF:
                bad[k] = vals
    return {"ok": not bad, "unexpected_diff": bad,
            "bias_keys": {a: {k: cfgs[a].get(k) for k in ("bias_wrap", "bias_off")} for a in have}}


def gen_line_ok():
    """(OP3)：主脚本与补跑器的 gen_trd 命令行必须逐字相同（只许 GPU 变量名不同）。"""
    out = {}
    for f in ("scripts/p8_bias_ablation.sh", "scripts/p8_retry.sh"):
        p = os.path.join(ROOT, f)
        if not os.path.exists(p):
            out[f] = None
            continue
        txt = open(p, encoding="utf-8").read().replace("\\\n", " ")
        line = next((l for l in txt.splitlines() if "eval/gen_trd.py" in l), None)
        if line is None:
            out[f] = None
            continue
        body = line.split("eval/gen_trd.py", 1)[1].split("||", 1)[0]
        out[f] = re.sub(r"\s+", " ", body).strip()
    vals = [v for v in out.values() if v]
    return {"ok": len(vals) == 2 and vals[0] == vals[1], "lines": out}


# ---------------------------------------------------------------- 主判据
def main_criterion(dev):
    """dev = {臂: {材质: |ratio−1|}} ⇒ 判据① 的逐臂读数与总判决。"""
    per_arm = {}
    for a in ABL:
        pair = sorted(set(dev[CTRL]) & set(dev[a]))
        d = [dev[a][m] - dev[CTRL][m] for m in pair]
        pos = sum(1 for x in d if x > 0)
        neg = sum(1 for x in d if x < 0)
        n = pos + neg
        med_a, med_c = median([dev[a][m] for m in pair]), median([dev[CTRL][m] for m in pair])
        p2 = binom_two_sided(pos, n)
        worse_median = bool(med_a > med_c + MARGIN)
        sig = bool(math.isfinite(p2) and p2 < ALPHA)
        per_arm[a] = {"n_paired": len(pair), "median_arm": med_a, "median_ctrl": med_c,
                      "delta_median": med_a - med_c, "worse_by_margin": worse_median,
                      "pos": pos, "neg": neg, "ties": len(pair) - n,
                      "p_two": p2, "p_one_sided_worse": binom_one_sided(pos, n),
                      "significant": sig, "arm_worse": bool(worse_median and sig),
                      "median_unpaired_arm": median(list(dev[a].values())),
                      "median_unpaired_ctrl": median(list(dev[CTRL].values()))}
    worse = [a for a in ABL if per_arm[a]["arm_worse"]]
    if len(worse) == len(ABL):
        verdict = "BIAS_IS_LOAD_BEARING"
    elif worse:
        verdict = "BIAS_PARTIAL"
    else:
        verdict = "BIAS_NULL"
    return verdict, worse, per_arm


def kid_register(ev):
    """判据②：只登记。对着 4.76 读，小于门槛一律记『什么也没测到』。"""
    if not ev:
        return None
    rows = {r.get("method"): r for r in ev} if isinstance(ev, list) else dict(ev)
    key = "p8%sx" % CTRL
    base = rows.get(key, {}).get("KID_x1e3")
    out = {"ctrl_method": key, "ctrl_KID": base, "thr": KID_THR, "arms": {}}
    for a in ABL + ["B2val", "v11dx"]:
        k = "p8%sx" % a if a in ABL else a
        v = rows.get(k, {}).get("KID_x1e3")
        if v is None or base is None:
            out["arms"][k] = {"KID_x1e3": v, "read": "NO_DATA"}
        else:
            d = v - base
            out["arms"][k] = {"KID_x1e3": v, "delta_vs_ctrl": d,
                              "read": "NOTHING_MEASURED" if abs(d) < KID_THR
                                      else ("WORSE" if d > 0 else "BETTER")}
    return out


# ---------------------------------------------------------------- selftest
def selftest():
    ok = 0
    fail = []

    def chk(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    # --- 统计工具
    chk("binom_two_sided(5,10)==1", abs(binom_two_sided(5, 10) - 1.0) < 1e-12)
    chk("binom_two_sided(10,10)", abs(binom_two_sided(10, 10) - 2.0 / 1024) < 1e-12)
    chk("binom_two_sided(0,10)", abs(binom_two_sided(0, 10) - 2.0 / 1024) < 1e-12)
    chk("binom_two_sided(8,10)", abs(binom_two_sided(8, 10) - 2 * (45 + 10 + 1) / 1024) < 1e-12)
    chk("binom_one_sided(10,10)", abs(binom_one_sided(10, 10) - 1.0 / 1024) < 1e-12)
    chk("binom_two_sided n=0 nan", math.isnan(binom_two_sided(0, 0)))
    chk("median even", abs(median([1.0, 3.0]) - 2.0) < 1e-12)
    chk("median odd", abs(median([5.0, 1.0, 3.0]) - 3.0) < 1e-12)
    chk("median empty nan", math.isnan(median([])))
    chk("finite drops nan", finite([1.0, float("nan")]) == [1.0])

    # --- 活件借用：tile_seam_ratio 的已知行为
    ns = activepieces()
    flat = np.full((16, 16), 7.0)
    chk("flat tile -> nan", math.isnan(ns["tile_seam_ratio"](flat)))
    rng = np.random.default_rng(0)
    noise = rng.uniform(0, 255, (16, 16))
    r_noise = ns["tile_seam_ratio"](noise)
    chk("noise ratio ~1", 0.7 < r_noise < 1.3)
    ramp = np.tile(np.arange(16, dtype=float) * 16, (16, 1))   # 左右不接缝的梯度
    chk("ramp ratio >> 1", ns["tile_seam_ratio"](ramp) > 5)
    chk("load_method borrowed", callable(ns["load_method"]))

    # --- 主判据逻辑：三臂全差 ⇒ LOAD_BEARING
    n = 40
    ctrl = {"m%02d" % i: 0.10 for i in range(n)}
    allworse = {a: {"m%02d" % i: 0.30 for i in range(n)} for a in ABL}
    v, worse, per = main_criterion({CTRL: ctrl, **allworse})
    chk("all worse -> LOAD_BEARING", v == "BIAS_IS_LOAD_BEARING" and worse == ABL)
    chk("delta_median 0.20", abs(per["A1"]["delta_median"] - 0.20) < 1e-12)
    chk("pos==n", per["A1"]["pos"] == n and per["A1"]["neg"] == 0)

    # --- 只有一条差 ⇒ PARTIAL（且照实报是哪条）
    mixed = {CTRL: ctrl,
             "A1": {k: 0.30 for k in ctrl},
             "A2": {k: 0.10 for k in ctrl},
             "A3": {k: 0.11 for k in ctrl}}      # 只高 0.01 < MARGIN ⇒ 不算差
    v, worse, per = main_criterion(mixed)
    chk("one worse -> PARTIAL", v == "BIAS_PARTIAL" and worse == ["A1"])
    chk("A3 margin not met", per["A3"]["significant"] and not per["A3"]["worse_by_margin"])

    # --- 边界：恰好高 MARGIN 不算过（判据写的是 >，⛔ 不许放宽成 >=）
    edge = {CTRL: ctrl, "A1": {k: 0.15 for k in ctrl},
            "A2": {k: 0.10 for k in ctrl}, "A3": {k: 0.10 for k in ctrl}}
    v, _, per = main_criterion(edge)
    chk("exact margin fails", (not per["A1"]["worse_by_margin"]) and v == "BIAS_NULL")

    # --- 没有一条差 ⇒ NULL；消融臂反而更好也是 NULL
    better = {CTRL: ctrl, "A1": {k: 0.02 for k in ctrl},
              "A2": {k: 0.02 for k in ctrl}, "A3": {k: 0.02 for k in ctrl}}
    v, worse, _ = main_criterion(better)
    chk("all better -> NULL", v == "BIAS_NULL" and worse == [])

    # --- 中位数过门槛但不显著 ⇒ 不算差（两条要求是「且」）
    few = {"m0": 0.10, "m1": 0.10, "m2": 0.10}
    nosig = {CTRL: few, "A1": {k: 0.90 for k in few},
             "A2": {k: 0.10 for k in few}, "A3": {k: 0.10 for k in few}}
    v, _, per = main_criterion(nosig)
    chk("margin w/o significance -> NULL",
        per["A1"]["worse_by_margin"] and not per["A1"]["significant"] and v == "BIAS_NULL")

    # --- 配对集：只在两臂都有读数的材质上算
    partial = {CTRL: {"a": 0.1, "b": 0.1}, "A1": {"a": 0.5},
               "A2": {"a": 0.1, "b": 0.1}, "A3": {"a": 0.1, "b": 0.1}}
    _, _, per = main_criterion(partial)
    chk("paired set only", per["A1"]["n_paired"] == 1)

    # --- 判据②：小于门槛一律 NOTHING_MEASURED
    ev = [{"method": "p8Cx", "KID_x1e3": 10.0}, {"method": "p8A1x", "KID_x1e3": 14.0},
          {"method": "p8A2x", "KID_x1e3": 20.0}, {"method": "p8A3x", "KID_x1e3": 3.0}]
    k = kid_register(ev)
    chk("KID below thr", k["arms"]["p8A1x"]["read"] == "NOTHING_MEASURED")
    chk("KID worse", k["arms"]["p8A2x"]["read"] == "WORSE")
    chk("KID better", k["arms"]["p8A3x"]["read"] == "BETTER")
    chk("KID missing", k["arms"]["B2val"]["read"] == "NO_DATA")

    # --- (OP1) 只允许 bias 两键不同
    base_cfg = {"steps": 12000, "lr": 1.5e-4, "bias_wrap": "torus", "bias_off": False, "out": "x"}
    good = {a: dict(base_cfg, out="p8" + a) for a in ARMS}
    good["A2"]["bias_wrap"] = "none"
    good["A1"]["bias_off"] = True
    chk("OP1 pass", config_diff(good)["ok"])
    bad = {a: dict(base_cfg) for a in ARMS}
    bad["A3"]["lr"] = 3e-4
    chk("OP1 catches lr", not config_diff(bad)["ok"] and "lr" in config_diff(bad)["unexpected_diff"])
    chk("OP1 missing cfg", not config_diff({CTRL: base_cfg})["ok"])

    # --- (OP3) 两个 .sh 的生成行逐字相同
    chk("OP3 gen lines identical", gen_line_ok()["ok"])

    print("selftest %d/%d" % (ok, ok + len(fail)) + ("" if not fail else "  FAIL: " + ", ".join(fail)))
    return not fail


# ---------------------------------------------------------------- 主流程
def analyse(base, runs, dry=False):
    ns = activepieces()
    present = {a: len(slugs_of(arm_dir(base, a))) for a in ARMS}
    rep = {"dir": base, "runs": runs, "stamp": STAMP, "materials_seen": present}
    if dry:
        rep["op"] = op_checks(base, runs)
        rep["op"]["OP3"] = gen_line_ok()
        return rep
    if any(v == 0 for v in present.values()):
        rep["verdict"] = "VOID_NO_DATA"
        rep["op"] = op_checks(base, runs)
        rep["op"]["OP3"] = gen_line_ok()
        return rep

    slugs = sorted(slugs_of(arm_dir(base, CTRL)))
    dev, diag = {}, {}
    for a in ARMS:
        dev[a], diag[a] = per_material_dev(base, a, slugs, ns)
    op = op_checks(base, runs)
    op["OP3"] = gen_line_ok()
    op_all = (op["OP1"]["ok"] and all(op["OP2"][a]["ok"] for a in ARMS)
              and all(op["OP4"][a]["ok"] for a in ARMS) and op["OP4"]["same_slugs"]
              and op["OP3"]["ok"])
    verdict, worse, per = main_criterion(dev)
    rep.update({"op": op, "op_all_pass": bool(op_all), "diag": diag,
                "per_arm": per, "arms_worse": worse,
                "verdict": verdict if op_all else "VOID_OP_FAILED",
                "verdict_if_op_ignored": verdict,
                "kid_register": kid_register(load_json(os.path.join(base, "p8_eval_Vmat_16.json")))})
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "remote_tmp", "p8"),
                    help="四臂 16px 产物：<dir>/p8{C,A1,A2,A3}x/16/*.png（要手动 scp，"
                         "远程在 /mnt/data 的 experiments/baselines/ 下，30 分钟同步拉不到）")
    ap.add_argument("--runs", default=os.path.join(ROOT, "remote_tmp", "runs"),
                    help="四臂训练目录：<runs>/trd_p8{C,A1,A2,A3}_<stamp>/（sync_remote_tmp.sh 自动拉）")
    ap.add_argument("--out")
    ap.add_argument("--dry", action="store_true", help="只查料在不在判读器眼里，不进主判据")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    rep = analyse(a.dir, a.runs, dry=a.dry)
    if a.out:                                    # ⛔ 落盘一律在打印之前（非 GBK 字形会崩控制台）
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
