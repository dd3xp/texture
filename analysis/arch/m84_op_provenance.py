#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""(M84) 盲期审计：七条作废条件里，**哪几条在判决日其实是旧料复读**。

为什么要在看到读数之前做这件事
--------------------------------
(M84) 的控制臂是**白拿**的 —— 28 份读数直接复用 (M75) 的 `m75_w384_s*.json`（同一条命令、
同一个 run）。好处已经记账；**代价**没记：凡是**只读控制臂**的操作检验，它的值在 (M75)
判决日就已经算过一次并已发表，(M84) 判决日再打印一遍**不携带任何关于本臂的新信息**。
若判决文本写「七条操作检验全过」，就会把**复读**当成**独立证据**。

本脚本做两件事（零 GPU、零 API、只读控制臂 ⇒ 对 ctrl-vs-trt 携 0 比特）：

  1. **机械地**测出每条检验的料来源（⛔ 不靠我读源码眼判）：给冻结的 `analyse()`
     喂合成料，分别只扰动 ctrl 一侧 / 只扰动 trt 一侧，看哪几条的输出跟着变。
     ⇒ 只随 ctrl 变的＝复读；随 trt 变的＝本臂新证据。
  2. 把真实 28 份控制臂读数喂给**冻结的** `analyse()` 里那几条 ctrl-only 检验，
     与 `experiments/m75_width.json` 已发表的值**逐位**比对，验证"复读"这句是事实而非推测。

⛔ 本脚本不改任何判据、不改冻结判读器、不落盘任何读数、不看处理臂（它此刻 0/28）。
⛔ 不是"提前退役"这几条：它们仍按 (M84) 预注册的固定顺序在判决日逐条跑。
   本脚本只决定**判决文本怎么写**（几条是新证据）。
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m84_read_retrain import (analyse, load_arm, REF_ROW_32, STEPS_EXPECT,
                              N_MAT_EXPECT, N_PERM)

OP_NAMES = ("OP1", "V4_nan", "OP2", "OP4a_steps", "OP4b_config",
            "OP5_codebook", "OP3_falsepos")
M75_JSON = "experiments/m75_width.json"


def mk(k, n_mat, shift, seed):
    """合成一条臂：k 个种子 × n_mat 个材质。"""
    rnd = random.Random(seed)
    mats = ["mat%02d" % i for i in range(n_mat)]
    return {s: {m: shift + rnd.gauss(0, 0.85) for m in mats} for s in range(k)}


def base_kwargs(k):
    cfg = {"out": "a", "seed": 0, "d": 384, "codebook_err": 1.25, "p32": 0.5}
    return dict(refs_ctrl={s: REF_ROW_32 for s in range(k)},
                refs_trt={s: REF_ROW_32 for s in range(k)},
                nans=[], cfg_c=dict(cfg),
                cfg_t=dict(cfg, out="b", seed=1),
                last_c=STEPS_EXPECT, last_t=STEPS_EXPECT)


def fingerprint(res):
    """把一次 analyse 的输出压成「每条检验一个可比对的指纹」。"""
    fp = {}
    for nm in OP_NAMES:
        fp[nm] = json.dumps(res.get("ops", {}).get(nm), sort_keys=True,
                            default=str)
    fp["MAIN"] = json.dumps(res.get("main"), sort_keys=True, default=str)
    return fp


def provenance(k=6, n_mat=N_MAT_EXPECT, n_perm=2000):
    """只扰动一侧，看哪几条检验跟着变 ⇒ 机械测出料来源。"""
    kw = base_kwargs(k)
    ctrl0 = mk(k, n_mat, 0.0, 11)
    trt0 = mk(k, n_mat, 0.0, 22)
    base = fingerprint(analyse(ctrl0, trt0, k, n_perm=n_perm, **kw))

    ctrl1 = mk(k, n_mat, 0.0, 33)          # 只换 ctrl 的料
    trt1 = mk(k, n_mat, 0.0, 44)           # 只换 trt 的料
    f_ctrl = fingerprint(analyse(ctrl1, trt0, k, n_perm=n_perm, **kw))
    f_trt = fingerprint(analyse(ctrl0, trt1, k, n_perm=n_perm, **kw))

    out = {}
    for nm in list(OP_NAMES) + ["MAIN"]:
        moved_c = f_ctrl[nm] != base[nm]
        moved_t = f_trt[nm] != base[nm]
        if moved_t:
            src = "both" if moved_c else "trt_only"
        else:
            src = "ctrl_only" if moved_c else "neither"
        out[nm] = {"moves_with_ctrl": moved_c, "moves_with_trt": moved_t,
                   "reads": src}
    return out


def provenance_meta(k=6, n_mat=N_MAT_EXPECT, n_perm=500):
    """第二道探针：扰动**处理臂那一侧的结构元数据**（份数/参照行/末步/config），
    看哪条检验因此触发。⚑ 第一道探针只扰动读数的**数值** ⇒ 会把这六条误标成
    "不读任何东西"；实际它们读的是 trt 侧的**元数据**＝本臂真·新证据。"""
    ctrl0 = mk(k, n_mat, 0.0, 11)
    trt0 = mk(k, n_mat, 0.0, 22)

    def verdict(**over):
        kw = base_kwargs(k)
        ctrl, trt = ctrl0, trt0
        if "drop_trt" in over:
            trt = {s: v for s, v in trt0.items() if s != 0}
        kw.update({kk: vv for kk, vv in over.items() if kk != "drop_trt"})
        return analyse(ctrl, trt, k, n_perm=n_perm, **kw)["verdict"]

    kwb = base_kwargs(k)
    cases = {
        "trt_missing_one_file": {"drop_trt": True},
        "trt_nan": {"nans": [3]},
        "trt_ref_row_wrong": {"refs_trt": {s: (1.0 if s == 2 else REF_ROW_32)
                                           for s in range(k)}},
        "trt_last_step_wrong": {"last_t": 4000},
        "trt_codebook_differs": {"cfg_t": dict(kwb["cfg_t"], codebook_err=9.9)},
        "trt_config_key_differs": {"cfg_t": dict(kwb["cfg_t"], p32=0.3)},
    }
    return {nm: verdict(**ov) for nm, ov in cases.items()}


def selftest():
    ok = 0
    prov = provenance()
    # 1) 阳性对照：主判据必须同时随两侧动
    assert prov["MAIN"]["reads"] == "both", prov["MAIN"]; ok += 1
    # 2) 阳性对照：(OP3) 必须随 ctrl 动（否则探针本身没在工作）
    assert prov["OP3_falsepos"]["moves_with_ctrl"], prov["OP3_falsepos"]; ok += 1
    # 3) 阴性对照：(OP3) ⛔ 不许随 trt 动（它是控制臂内部检验）
    assert not prov["OP3_falsepos"]["moves_with_trt"], prov["OP3_falsepos"]; ok += 1
    assert prov["OP3_falsepos"]["reads"] == "ctrl_only"; ok += 1
    # 4) 只读 config/参照行/末步 的四条：两侧读数都不影响
    for nm in ("OP1", "V4_nan", "OP2", "OP4a_steps", "OP4b_config", "OP5_codebook"):
        assert prov[nm]["reads"] == "neither", (nm, prov[nm]); ok += 1
    # 5) 指纹函数本身要能分辨（阳性对照：同输入必须同指纹）
    kw = base_kwargs(6)
    c, t = mk(6, N_MAT_EXPECT, 0.0, 11), mk(6, N_MAT_EXPECT, 0.0, 22)
    f1 = fingerprint(analyse(c, t, 6, n_perm=500, **kw))
    f2 = fingerprint(analyse(c, t, 6, n_perm=500, **kw))
    assert f1 == f2; ok += 1
    # 6) 第二道探针：扰动 trt 侧元数据必须逐条打出对应的 VOID（否则"六条不读任何东西"是错的）
    meta = provenance_meta()
    expect = {"trt_missing_one_file": "VOID_NO_DATA",
              "trt_nan": "VOID_NAN",
              "trt_ref_row_wrong": "VOID_REF_ROW",
              "trt_last_step_wrong": "VOID_UNDERTRAINED",
              "trt_codebook_differs": "VOID_DATA_MISMATCH",
              "trt_config_key_differs": "VOID_CONFIG_DIFF"}
    for nm, want in expect.items():
        assert meta[nm] == want, (nm, meta[nm], want); ok += 1
    print("selftest OK  (%d/%d)" % (ok, ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="remote_tmp")
    ap.add_argument("--ctrl_tag", default="m75_w384")
    ap.add_argument("--k", type=int, default=28)
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return

    res = {"provenance": provenance(), "provenance_meta": provenance_meta()}

    # ---- 真实控制臂：把 ctrl-only 那条检验按冻结实现算出来，与 (M75) 已发表值逐位比 ----
    ctrl, ctrl_rows, probs = load_arm(a.dir, a.ctrl_tag, a.k)
    mats = sorted(set.intersection(*[set(v) for v in ctrl.values()])) if ctrl else []
    kw = base_kwargs(a.k)
    kw["refs_ctrl"] = {s: REF_ROW_32 for s in ctrl}
    kw["refs_trt"] = {s: REF_ROW_32 for s in ctrl}
    # 用 ctrl 自己充当两侧只为让冻结 analyse() 跑到 (OP3) 那一步；
    # ⛔ 主判据的输出在这里无意义（ctrl vs ctrl 恒 0），不读、不落盘。
    full = analyse(ctrl, ctrl, a.k, n_perm=N_PERM, **kw)
    op3 = full.get("ops", {}).get("OP3_falsepos")
    res["ctrl_only_recomputed"] = {
        "n_ctrl": len(ctrl), "n_mat": len(mats), "load_problems": probs,
        "OP3": None if op3 is None else {k2: op3[k2] for k2 in ("ok", "p", "mean")},
    }

    pub = None
    if os.path.exists(M75_JSON):
        with open(M75_JSON, encoding="utf-8") as f:
            m75 = json.load(f)
        o = m75.get("ops", {}).get("OP3_falsepos", {})
        pub = {"p": o.get("p"), "mean": o.get("mean")}
    res["m75_published_OP3"] = pub
    res["OP3_bit_identical"] = bool(
        pub and op3 and repr(pub["p"]) == repr(op3["p"])
        and repr(pub["mean"]) == repr(op3["mean"]))
    new_ev = [nm for nm in OP_NAMES if res["provenance"][nm]["reads"] in ("trt_only", "both")]
    recycled = [nm for nm in OP_NAMES if res["provenance"][nm]["reads"] == "ctrl_only"]
    fires_on_trt_meta = sorted(set(res["provenance_meta"].values()))
    res["summary"] = {
        "n_checks": len(OP_NAMES),
        "recycled_from_m75_ctrl_only": recycled,
        "moves_with_trt_readings_values": new_ev,
        "insensitive_to_reading_values": [nm for nm in OP_NAMES
                                          if res["provenance"][nm]["reads"] == "neither"],
        "but_fire_on_trt_metadata": fires_on_trt_meta,
        "note": ("六条对读数**数值**不敏感，但逐条能被处理臂**元数据**触发 ⇒ 它们是本臂真·新证据；"
                 "只有 (OP3) 是 (M75) 旧料复读"),
    }

    if a.out:                                   # 落盘一律在打印之前
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)

    print("== (M84) 七条作废条件的料来源（机械测定，非眼判）==")
    for nm in list(OP_NAMES) + ["MAIN"]:
        p = res["provenance"][nm]
        print("  %-13s reads=%-9s (ctrl:%s trt:%s)"
              % (nm, p["reads"], p["moves_with_ctrl"], p["moves_with_trt"]))
    s = res["summary"]
    print("ctrl-only(=(M75) 旧料复读): %s" % ", ".join(s["recycled_from_m75_ctrl_only"]))
    print("对读数数值不敏感: %s" % ", ".join(s["insensitive_to_reading_values"]))
    print("== 第二道探针：扰动处理臂元数据，逐条应打出自己那条 VOID ==")
    for nm, v in res["provenance_meta"].items():
        print("  %-24s -> %s" % (nm, v))
    r = res["ctrl_only_recomputed"]
    print("控制臂重算 (OP3): n=%s n_mat=%s p=%s mean=%s"
          % (r["n_ctrl"], r["n_mat"],
             None if not r["OP3"] else r["OP3"]["p"],
             None if not r["OP3"] else r["OP3"]["mean"]))
    print("(M75) 已发表 (OP3): %s" % (pub,))
    print("逐位相同: %s" % res["OP3_bit_identical"])
    print("【禁】本脚本不改判据、不退役任何一条；只决定判决文本里哪几条算新证据")


if __name__ == "__main__":
    main()
