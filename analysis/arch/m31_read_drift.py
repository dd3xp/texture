"""(M31) 判读器：把 `scripts/m31_drift_calib.sh` 头部写死的判据 (1)(2)(3)(5) 机械执行一遍。

===== 为什么要有这个脚本 =====
判据是跑前写死的，但**读数是手抄的**。本项目已经在这上面栽过两次：
  - `run_eval.py` 写的键名是 `KID_x1e3` 而不是 `KID`，取错键会拿到 `None`，
    而 `None` 与"不过门"在同一个布尔里长得一样（(M30) 记下的第三条硬规矩）;
  - `run_eval.py:100` 的 `ok` 是**每个方法各算各的**，文档里那句"各方法 n 相同"
    只在覆盖全满时成立 —— 若某条臂缺材质，三条臂的 KID 就不在同一个 n 上，
    而 (M31) 的预注册没覆盖这一条。
所以把判据写成代码，读数与判决一起落成产物（`experiments/m31_drift.json`）。

⚑ **本文件写于 seed2 还在训练时**（seed2 的 16px KID 当时尚不存在、v11dx 的 6.198 是旧读数）
  —— 判读器对结果是盲的。⛔ 出结果后**一个字都不许改判据**。

===== 逐条照抄 `scripts/m31_drift_calib.sh`（⛔ 不许放宽）=====
(OP1) 三条臂的 `config.json` 逐键比对，**只允许 `seed` 这一个键不同**；
      `out` 是产物路径不是配方，同样豁免；
      后加的默认键 `bias_cells=[] / p_tile16=0.0 / pack_balance=0.0 / bias_size_cond=False`
      视为相同（控制臂 v11d 跑在这些参数加进来之前，config 里根本没有这些键）。
      ⚠ 本轮 (M32) 新加的 `reseed_after_build` 也属于这一类，但**只有在它等于默认值 False 时**才豁免 ——
      若某条臂真的开了它，那就是配方不同，必须判 OPS_FAILED。
(OP2) 三条臂都跑到 12000 步且 `last.pt` 存在。
(OP3) 生成/评测口径一致 —— 本脚本能查的是：三条臂在同一 size 上的 `n` 与 `materials` 相等。
(2)   主统计量 = 三条臂两两之间 **16px KID** 的绝对差（seed0-1、seed0-2、seed1-2 共 3 个），
      Dmax = max|dKID|。⚠ n=3 个差值，**只报区间不做检验**（⛔ 不许对 3 个数算 p 值）。
(3)   Dmax >= 1.0 -> `DRIFT_EXCEEDS_GATE`；Dmax < 1.0 -> `DRIFT_WITHIN_GATE`。
(5)   次要报告项：16px 的 FID/FD/CLIP 与 24/32px 的 KID/FID/FD，各自的漂移一并报出。
      ⛔ 不授权任何事、⛔ 不许拿来给任何配置排序。

用法（在**远程**跑，因为 config 在 /mnt/data、eval JSON 在 /tmp；纯标准库、零 GPU）：
  python analysis/arch/m31_read_drift.py \
      --eval16 /tmp/eval_drift_Vmat_16.json \
      --eval24 /tmp/eval_drift_Vmat_24.json \
      --eval32 /tmp/eval_drift_Vmat_32.json \
      --out /tmp/m31_drift.json
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

GATE = 1.0                      # 判据 (3) 的门槛，⛔ 一个字不许放宽

# 三条臂：(名字, 默认 run 目录, 16px 方法名, 24/32px 方法名)
ARMS = [
    ("seed0", ROOT / "runs/trd_v11d", "v11dx", "v11dx_direct"),
    ("seed1", Path("/tmp/runs/trd_seed1_09161230"), "seed1x", "seed1x_direct"),
    ("seed2", Path("/tmp/runs/trd_seed2_09161230"), "seed2x", "seed2x_direct"),
]

# (OP1) 豁免键 -> 只有取这个值时才豁免；None 表示任意值都豁免（产物路径 / 唯一变量）
GRANDFATHERED = {
    "bias_cells": [],
    "p_tile16": 0.0,
    "pack_balance": 0.0,
    "bias_size_cond": False,
    "reseed_after_build": False,
}
FREE_KEYS = {"out", "seed"}     # out = 产物路径；seed = 本轮唯一变量


def op1(cfgs):
    """逐键比对。返回 (通过?, 问题列表, 每条臂的 seed)。"""
    bad, seeds = [], {}
    keys = set()
    for c in cfgs.values():
        keys |= set(c)
    for name, c in cfgs.items():
        seeds[name] = c.get("seed", 0)       # v11d 跑在 --seed 加进来之前，那一行原本写死 0
    for k in sorted(keys - FREE_KEYS):
        vals = {}
        for name, c in cfgs.items():
            v = c[k] if k in c else GRANDFATHERED.get(k, "<缺键>")
            vals[name] = v
        uniq = {json.dumps(v, sort_keys=True) for v in vals.values()}
        if len(uniq) > 1:
            bad.append(f"{k}: " + ", ".join(f"{n}={v!r}" for n, v in vals.items()))
    return (not bad), bad, seeds


def op2(runs):
    """跑满 12000 步 + last.pt 存在。"""
    bad = []
    for name, run in runs.items():
        if not (run / "last.pt").exists():
            bad.append(f"{name}: 缺 {run}/last.pt")
        log = run / "log.json"
        if not log.exists():
            bad.append(f"{name}: 缺 {log}")
            continue
        recs = json.loads(log.read_text(encoding="utf-8"))
        if isinstance(recs, dict):
            recs = recs.get("log", [])
        last = max((r.get("step", -1) for r in recs), default=-1)
        if last != 12000:
            bad.append(f"{name}: log.json 最后一步 {last} != 12000")
    return (not bad), bad


def op3(tbl16, tbl):
    """三条臂在同一 size 上的 n 与 materials 相等（预注册没写死，但不等就不在同一把尺子上）。

    ⚠ 返回 `checked` = 实际查了几个 size。**查了 0 个不叫通过**——eval JSON 还没生成时
    `bad` 天然是空的，若直接读成"通过"，就是拿"没量"冒充"量过且没事"。
    """
    bad, checked = [], []
    for size, t in [(16, tbl16)] + [(s, x) for s, x in tbl.items()]:
        if t is None:
            continue
        checked.append(size)
        got = {}
        for name, _run, m16, mdir in ARMS:
            key = m16 if size == 16 else mdir
            if key not in t:
                bad.append(f"{size}px: eval JSON 里没有 {key}")
                continue
            got[name] = (t[key].get("n"), t[key].get("materials"))
        if len(set(got.values())) > 1:
            bad.append(f"{size}px: 各臂 (n, materials) 不等 -> {got}")
    return (not bad and 16 in checked), bad, checked


def pairs(vals):
    """三条臂两两之差的绝对值（seed0-1、seed0-2、seed1-2）。"""
    out = []
    names = [a[0] for a in ARMS]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if vals.get(a) is None or vals.get(b) is None:
                continue
            out.append((f"{a}-{b}", abs(vals[a] - vals[b])))
    return out


def column(tbl, size, key):
    """取某个指标在三条臂上的值。键名已在真 JSON 上验过：KID 是 `KID_x1e3`。"""
    vals = {}
    for name, _run, m16, mdir in ARMS:
        m = m16 if size == 16 else mdir
        vals[name] = tbl.get(m, {}).get(key) if tbl else None
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval16", type=Path, default=Path("/tmp/eval_drift_Vmat_16.json"))
    ap.add_argument("--eval24", type=Path, default=Path("/tmp/eval_drift_Vmat_24.json"))
    ap.add_argument("--eval32", type=Path, default=Path("/tmp/eval_drift_Vmat_32.json"))
    ap.add_argument("--out", type=Path, default=Path("/tmp/m31_drift.json"))
    ap.add_argument("--runs", nargs=3, type=Path, default=None,
                    help="三条臂的 run 目录（seed0 seed1 seed2）；默认见 ARMS。只为自测留的口子")
    a = ap.parse_args()

    runs = {n: d for n, d, _, _ in ARMS}
    if a.runs:
        runs = {n: d for (n, _, _, _), d in zip(ARMS, a.runs)}

    res = {"gate": GATE, "arms": [x[0] for x in ARMS]}

    # ---- (1) 操作检验 ----
    cfg_paths = {n: d / "config.json" for n, d in runs.items()}
    missing = [f"{n}: 缺 {p}" for n, p in cfg_paths.items() if not p.exists()]
    if missing:
        print("OPS_FAILED（config 不全）：")
        for m in missing:
            print("  " + m)
        res["verdict"] = "OPS_FAILED"
        res["ops"] = {"missing_config": missing}
        a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        return

    cfgs = {n: json.loads(p.read_text(encoding="utf-8")) for n, p in cfg_paths.items()}
    ok1, bad1, seeds = op1(cfgs)
    ok2, bad2 = op2(runs)

    tbl16 = json.loads(a.eval16.read_text(encoding="utf-8")) if a.eval16.exists() else None
    tbl = {}
    for s, p in [(24, a.eval24), (32, a.eval32)]:
        tbl[s] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    ok3, bad3, checked3 = op3(tbl16, tbl)

    print(f"(OP1) 配方逐键比对：{'通过' if ok1 else '不通过'}；seed = {seeds}")
    for b in bad1:
        print("  差异 " + b)
    print(f"(OP2) 12000 步 + last.pt：{'通过' if ok2 else '不通过'}")
    for b in bad2:
        print("  " + b)
    how3 = "通过" if ok3 else ("不通过" if bad3 else f"未检查（只查到 {checked3 or '零'} 个 size 的 eval JSON）")
    print(f"(OP3) 各臂 n/materials 相等：{how3}（已查 {checked3}）")
    for b in bad3:
        print("  " + b)

    res["ops"] = {"op1": ok1, "op1_diffs": bad1, "op2": ok2, "op2_problems": bad2,
                  "op3": ok3, "op3_problems": bad3, "op3_sizes_checked": checked3,
                  "seeds": seeds}

    if not (ok1 and ok2 and ok3):
        print("\n=> OPS_FAILED，按预注册 (1) 不下判决。")
        res["verdict"] = "OPS_FAILED"
        a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        return

    if tbl16 is None:
        print("\n16px 的 eval JSON 还不存在 -> 无法算主统计量。")
        res["verdict"] = "PENDING"
        a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        return

    # ---- (2) 主统计量：16px KID ----
    kid = column(tbl16, 16, "KID_x1e3")
    if any(v is None for v in kid.values()):
        print(f"\n16px KID 缺读数：{kid} -> 不下判决（【禁】缺键不等于过门）")
        res["verdict"] = "OPS_FAILED"
        res["kid16"] = kid
        a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        return

    dif = pairs(kid)
    dmax = max(d for _, d in dif)
    dmin = min(d for _, d in dif)
    verdict = "DRIFT_EXCEEDS_GATE" if dmax >= GATE else "DRIFT_WITHIN_GATE"
    res.update({"kid16": kid, "pairwise_abs_diff_kid16": dict(dif),
                "Dmax": dmax, "Dmin": dmin, "verdict": verdict})
    # 先落盘再打印：打印崩了也不丢判决（Windows 控制台的字形问题本项目已犯过三次）
    a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")

    print("\n== 判据 (2) 主统计量：16px KID_x1e3 ==")
    for n, v in kid.items():
        print(f"  {n:<6} {v:.3f}")
    print("  两两绝对差：" + "，".join(f"{n} {d:.3f}" for n, d in dif))
    print(f"  Dmax = {dmax:.3f}（3 个差值区间 [{dmin:.3f}, {dmax:.3f}]；【禁】n=3，不做检验、无 p 值）")

    print(f"\n=> 判 {verdict}（门槛 {GATE}）")
    if verdict == "DRIFT_EXCEEDS_GATE":
        print(f"   今后守门必须画在 max(Dmax={dmax:.3f}, 采样下限 4.8) = {max(dmax, 4.8):.3f} 之外，"
              f"或改用 (M32) 的成对配方（--init_from + --reseed_after_build）。")
    else:
        print("   +-1.0 的守门落在漂移之外，今后可续用（仍受 m=1 采样下限 4.8 约束）。")
    print("【禁】判据 (4)：无论 Dmax 多大，(M29) 的 GATE_FAILED 一个字不改。")

    # ---- (5) 次要报告项 ----
    print("\n== 判据 (5) 次要报告项（【禁】不授权任何事、不许用来排序）==")
    sec = {}
    for size, t in [(16, tbl16), (24, tbl[24]), (32, tbl[32])]:
        if t is None:
            continue
        keys = ["KID_x1e3", "FID", "FD_DINOv2", "CLIP"] if size != 16 else ["FID", "FD_DINOv2", "CLIP"]
        for k in keys:
            vals = column(t, size, k)
            if any(v is None for v in vals.values()):
                print(f"  {size}px {k:<10} 缺读数 {vals}")
                continue
            d = pairs(vals)
            sec[f"{size}px_{k}"] = {"values": vals, "max_abs_diff": max(x for _, x in d)}
            print(f"  {size}px {k:<10} " + "  ".join(f"{n}={v:.3f}" for n, v in vals.items())
                  + f"  | 最大漂移 {max(x for _, x in d):.3f}")
    res["secondary"] = sec

    a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
