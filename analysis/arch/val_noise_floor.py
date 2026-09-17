#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""val 曲线自己的噪声下限，以及 (M41) 预测 (P3) 的判读门（零 API、零 GPU）。

为什么要有它：(M41) 预注册里写死了 (P3)「A（带 CLIP 损失）的 16px val 会比 B 差」。
但账本从来没量过 **val 这条曲线自己能抖多少** —— 而 (M37) 已经证明 val 在挑检查点这件事上
指错过方向。没有噪声下限，"A 的 val 高了 0.003" 这种话既不能算验证也不能算推翻。

做法（和 (M33) 给 KID/FID 定门槛的办法同一套）：拿一对**同配方、只换随机种子**的重训，
在**对齐的 step** 上算 val 之差，取 **Dmax**（去掉 step 0）当门槛。项目现有的唯一这样一对是
`runs/trd_seed1_09161230` / `runs/trd_seed2_09161230`。

⛔ 四条使用纪律（写进代码，免得以后被当成别的东西用）：
 1. ⛔ **val 不是任何一条判据**（(M41) 预注册、(M37) 实测它指错方向）。这里只为把 (P3)
    这条预测**照录**得诚实，⛔ 不许拿本文件的输出去挑检查点、挑配置、或推翻/支持判官。
 2. ⚠ 参照那一对的配方**与 (M41) 两臂不同**（16+32 混训 / 12000 步 vs 16px-only / 6000 步），
    所以门槛只是**同量级参照**，⛔ 不许说成"精确的 val 噪声"。偏差方向未知。
 3. ⚠ 参照那一对**换了种子**，而 (M41) 两臂是**成对配方**（同种子、同取批顺序）=>
    成对臂之间的真实噪声只会**更小**。⇒ 门槛偏保守：它防的是"把噪声读成效应"，
    ⛔ 代价是可能把小的真差别记成"没测到" —— 这两句必须一起引。
 4. ⛔ 项目**没有**量过"同配方同种子重训两次"的 val 差（那才是成对臂的真下限）。
    ⛔ 不许因为 step 0 的两臂只差 1e-4 就说门槛应该是 1e-4：step 0 还没有任何训练分歧。
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# 参照对（同配方、只换种子）与 (M41) 两臂的默认位置。
REF_A = ROOT / "remote_tmp" / "runs" / "trd_seed1_09161230"
REF_B = ROOT / "remote_tmp" / "runs" / "trd_seed2_09161230"
ARM_A = ROOT / "remote_tmp" / "runs" / "trd_clipw_09162316"   # 带 CLIP 损失
ARM_B = ROOT / "remote_tmp" / "runs" / "trd_ctl_09162316"     # 控制臂

CFG_KEYS = ["seed", "sizes", "steps", "batch", "lr", "p32", "init_from",
            "clip_loss", "clip_w", "clip_bs"]


def read_log(d: Path):
    """{step: val}；缺文件返回 None（⛔ 不许把缺数据冒充成空集"没事"）。"""
    p = d / "log.json"
    if not p.exists():
        return None
    rows = json.loads(p.read_text(encoding="utf-8"))
    return {int(r["step"]): float(r["val"]) for r in rows
            if r.get("val") is not None}


def read_cfg(d: Path):
    p = d / "config.json"
    if not p.exists():
        return None
    c = json.loads(p.read_text(encoding="utf-8"))
    return {k: c.get(k) for k in CFG_KEYS}


def diffs(la, lb, drop_step0=True):
    """对齐 step 上的 val_a - val_b。"""
    ks = sorted(set(la) & set(lb))
    if drop_step0:
        ks = [k for k in ks if k != 0]
    return [(k, la[k] - lb[k]) for k in ks]


def floor_from(la, lb):
    """(M33) 的惯例：门槛 = Dmax（对齐 step 上差的绝对值最大者），均值只当次要读数。"""
    ds = diffs(la, lb)
    if not ds:
        return None
    a = [abs(d) for _, d in ds]
    return {"n_steps": len(ds), "dmax": max(a), "dmean": sum(a) / len(a),
            "per_step": [{"step": k, "d": d} for k, d in ds]}


def verdict(d_final, thr):
    """val 是损失，越低越好 => A 更差 = d = val_A - val_B 为正且过门。"""
    if d_final >= thr:
        return "VAL_P3_CONFIRMED"      # A 的 val 确实更差
    if d_final <= -thr:
        return "VAL_P3_REFUTED"        # A 的 val 反而更好
    return "VAL_BELOW_NOISE"           # 没测到（⛔ 不是"打平"）


def run(args):
    out = {"problems": [], "checked": 0}

    # --- 一、门槛：同配方只换种子的那一对 ---
    out["checked"] += 1
    ra, rb = read_log(Path(args.ref_a)), read_log(Path(args.ref_b))
    if ra is None or rb is None:
        out["problems"].append("缺参照对 log.json: %s / %s" % (args.ref_a, args.ref_b))
        out["floor"] = None
        out["thr"] = None
    else:
        out["floor"] = floor_from(ra, rb)
        out["thr"] = out["floor"]["dmax"] if out["floor"] else None
        out["ref_cfg"] = {"a": read_cfg(Path(args.ref_a)), "b": read_cfg(Path(args.ref_b))}

    # --- 二、两臂：对齐 step 上的差 + 判读 ---
    out["checked"] += 1
    aa, ab = read_log(Path(args.arm_a)), read_log(Path(args.arm_b))
    if aa is None or ab is None:
        out["problems"].append("缺两臂 log.json: %s / %s" % (args.arm_a, args.arm_b))
        out["arms"] = None
        out["verdict"] = "VOID_NO_DATA"
    else:
        ds = diffs(aa, ab, drop_step0=False)
        done = max(aa) >= args.steps and max(ab) >= args.steps
        out["arms"] = {
            "steps_a": max(aa), "steps_b": max(ab), "complete": done,
            "per_step": [{"step": k, "val_a": aa[k], "val_b": ab[k], "d": d}
                         for k, d in ds],
            "cfg": {"a": read_cfg(Path(args.arm_a)), "b": read_cfg(Path(args.arm_b))},
        }
        if not done:
            out["problems"].append("两臂未跑完（A %d / B %d，需 %d 步）=> 不下判读"
                                   % (max(aa), max(ab), args.steps))
            out["verdict"] = "VOID_NOT_DONE"
        elif out["thr"] is None:
            out["verdict"] = "VOID_NO_DATA"
        else:
            d_fin = aa[args.steps] - ab[args.steps]
            out["d_final"] = d_fin
            out["verdict"] = verdict(d_fin, out["thr"])

    # 落盘一律挪到打印之前（(M33) 的规矩：崩了数据也在）。
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                              encoding="utf-8")

    f = out.get("floor")
    if f:
        print("[门槛] 参照对 %s vs %s：对齐 %d 个 step（已去 step0）"
              % (Path(args.ref_a).name, Path(args.ref_b).name, f["n_steps"]))
        print("       Dmax = %.4f（门槛）， 平均 |d| = %.4f（次要读数）"
              % (f["dmax"], f["dmean"]))
    else:
        print("[门槛] 【禁】没量到：参照对数据缺失")
    a = out.get("arms")
    if a:
        print("[两臂] A=%s(%d 步) B=%s(%d 步)"
              % (Path(args.arm_a).name, a["steps_a"], Path(args.arm_b).name, a["steps_b"]))
        for r in a["per_step"]:
            print("       step %6d  val_A %.4f  val_B %.4f  d %+.5f"
                  % (r["step"], r["val_a"], r["val_b"], r["d"]))
    else:
        print("[两臂] 【禁】没量到：两臂 log.json 缺失")
    if "d_final" in out:
        print("[判读] d(final) = %+.5f  门槛 %.4f" % (out["d_final"], out["thr"]))
    print("[判读] %s" % out["verdict"])
    print("已查 %d 项，problems=%d" % (out["checked"], len(out["problems"])))
    for p in out["problems"]:
        print("  - " + p)
    print("写入 " + str(args.out))
    return 0 if not out["problems"] else 1


def selftest():
    """合成日志验算 Dmax / 三种判读；⛔ 不碰真数据。"""
    tmp = Path(tempfile.mkdtemp())

    def mk(name, vals):
        d = tmp / name
        d.mkdir()
        (d / "log.json").write_text(json.dumps(
            [{"step": s, "val": v} for s, v in vals]), encoding="utf-8")
        return d

    # 参照对：step0 差 9（应被丢掉），其余差 0.01 / -0.05 / 0.02 => Dmax 0.05
    ra = mk("ra", [(0, 10.0), (1, 1.01), (2, 1.00), (3, 1.02)])
    rb = mk("rb", [(0, 1.0), (1, 1.00), (2, 1.05), (3, 1.00)])
    f = floor_from(read_log(ra), read_log(rb))
    assert f["n_steps"] == 3, f
    assert abs(f["dmax"] - 0.05) < 1e-9, f
    assert abs(f["dmean"] - (0.01 + 0.05 + 0.02) / 3) < 1e-9, f
    assert verdict(+0.05, 0.05) == "VAL_P3_CONFIRMED"
    assert verdict(-0.05, 0.05) == "VAL_P3_REFUTED"
    assert verdict(+0.0499, 0.05) == "VAL_BELOW_NOISE"
    assert verdict(-0.0499, 0.05) == "VAL_BELOW_NOISE"
    assert read_log(tmp / "nope") is None            # 缺数据必须报 None
    print("selftest ok")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_a", default=str(REF_A))
    ap.add_argument("--ref_b", default=str(REF_B))
    ap.add_argument("--arm_a", default=str(ARM_A))
    ap.add_argument("--arm_b", default=str(ARM_B))
    ap.add_argument("--steps", type=int, default=6000, help="两臂的终点 step")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "m41_val_noise.json"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    return selftest() if args.selftest else run(args)


if __name__ == "__main__":
    sys.exit(main())
