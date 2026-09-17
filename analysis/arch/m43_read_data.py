#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""(M43) 判读器 —— **盲写于任何一张图存在之前**。判据不是本文件定的，全文在
`eval/m43_data_ab.sh` 头部（与本文件同一次提交 = 预注册）。本文件只是把那些话逐字翻成代码。

问题：2026-09-11 那次把 **80% 的补充训练数据**（`train_extra.json` 12241 张 ->
`train_extra_packs_only.json` 2433 张）扔掉的选择，第一次交给现行判官。
  A = data7 = `runs/trd_v7` step_20000.pt（含模组训练池 + `--domain`）
  B = data8 = `runs/trd_v8` step_20000.pt（= 现行 16px 主配置的权重）
⚠ 不是干净的数据臂（见预注册 (2)）⇒ 只授权"采纳"层面的话。

两种用法（同一份判据，分两个时刻执行）：
  python analysis/arch/m43_read_data.py --selftest          # 先自测，必须先过
  python analysis/arch/m43_read_data.py screen --root /tmp/m43ab --ra runs/trd_v7 \
      --rb runs/trd_v8 --out /tmp/m43_meter.json            # 远程，零 API：操作检验 + 筛子
  python analysis/arch/m43_read_data.py judge               # 本机，判官 JSON 到手之后

⚠ 打印里只用 ASCII 与常见汉字（Windows 控制台 GBK），非 GBK 字形只许出现在注释里；
⚠ 落盘一律在打印之前；⚠ 本机读远程 JSON 一律 encoding="utf-8"。
"""
import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

SIZE = "16"
D37, D38, D40 = 0.8173, 0.6608, 0.6431      # ⛔ 三锚点写死，一个字不许改
D_SEED = 0.6643                              # (M42) 构造性零对照，只登记、不是门
WANT_STEP = 20000
CFG_ALLOWED = {"out", "save_at", "steps", "domain", "extra_file", "codebook_err"}
CFG_REQUIRED = {"domain", "extra_file"}
SAME_MAX = 25                                # (OP5)：逐像素相同的材质上限（(M37)(M38)(M40) 同一道门）
ARM = "judge_full_data7_rr4_vs_data8_rr4_16_V_mat"
EXP = ROOT / "experiments"
OUT_JUDGE = EXP / "m43_data_cluster.json"
FLOOR = 21 / 118


# --------------------------------------------------------------------------- 尺子
def _stats(pairs):
    """pairs: [(name, arrA, arrB)] -> 逐像素差别统计（与 (M39)(M41)(M42) 的尺子逐字相同）。"""
    import numpy as np
    if not pairs:
        return {"n": 0, "identical": 0, "pixfrac": None, "mae": None}
    fr, ae, ident = [], [], 0
    for _, a, b in pairs:
        a = a.astype(np.int32)
        b = b.astype(np.int32)
        d = np.abs(a - b).reshape(-1, 3)
        nd = int((d.sum(axis=-1) > 0).sum())
        fr.append(nd / d.shape[0])
        ae.append(float(d.mean()) / 255.0)
        if nd == 0:
            ident += 1
    return {"n": len(pairs), "identical": ident,
            "pixfrac": float(sum(fr) / len(fr)), "mae": float(sum(ae) / len(ae))}


def screen_of(d):
    """预注册 (5) 的门。⚠ 闭区间边界：>=D37 算 GO、<=D38 算 NOGO。"""
    if d >= D37:
        return "SCREEN_GO"
    if d <= D38:
        return "SCREEN_NOGO"
    return "SCREEN_AMBIGUOUS"


def opcheck(counts, cfg_a, cfg_b, md5_a, md5_b, step_a, step_b, identical):
    """预注册 (4) 的 (OP1)-(OP5)。返回 (problems, checked)。"""
    bad, checked = [], 0
    for tag, want in (("data7", 500), ("data8", 500), ("data7_rr4", 125), ("data8_rr4", 125)):
        checked += 1
        if counts.get(tag) != want:
            bad.append("OP1:%s=%s" % (tag, counts.get(tag)))
    checked += 1
    diff = set(k for k in set(cfg_a) | set(cfg_b) if cfg_a.get(k) != cfg_b.get(k))
    if diff - CFG_ALLOWED:
        bad.append("OP2:extra_keys=%s" % (sorted(diff - CFG_ALLOWED),))
    if CFG_REQUIRED - diff:
        bad.append("OP2:missing=%s" % (sorted(CFG_REQUIRED - diff),))
    checked += 1
    if not (step_a == step_b == WANT_STEP):
        bad.append("OP3:steps=%s/%s" % (step_a, step_b))
    checked += 1
    if md5_a == md5_b:                       # 正向对照：码本由训练数据建，必须**不同**
        bad.append("OP4:codebook_md5_same")
    checked += 1
    if identical is None:
        bad.append("OP5:no_data")
    elif identical > SAME_MAX:
        bad.append("OP5:same=%d" % identical)
    return bad, checked


# --------------------------------------------------------------------------- 判官阶段
def decide(resolvable, cluster_verdict, lo, hi):
    """预注册 (6) 的四种读法，逐字映射；事后不许再编第五种。"""
    if not resolvable:
        return "VOID_UNRESOLVABLE"
    if cluster_verdict.startswith("VOID"):
        return "VOID_" + cluster_verdict
    if lo > 0.5:
        return "DATA_V7_WINS"
    if hi < 0.5:
        return "DATA_V8_WINS"
    return "DATA_NULL"


def _md5(p):
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


def _ckpt_step(run, name="step_20000.pt"):
    import torch
    return int(torch.load(pathlib.Path(run) / name, map_location="cpu")["step"])


def _load_pairs(da, db, problems):
    import numpy as np
    from PIL import Image
    da, db = pathlib.Path(da), pathlib.Path(db)
    if not da.is_dir() or not db.is_dir():
        problems.append("MISSING_DIR:%s" % (da if not da.is_dir() else db))
        return []
    out = []
    for f in sorted(da.glob("*.png")):
        g = db / f.name
        if not g.exists():
            problems.append("MISSING_FILE:%s" % f.name)
            continue
        x = np.asarray(Image.open(f).convert("RGB"))
        y = np.asarray(Image.open(g).convert("RGB"))
        if x.shape != y.shape:
            problems.append("SHAPE:%s" % f.name)
            continue
        out.append((f.name, x, y))
    return out


def run_screen(a):
    root, ra, rb = pathlib.Path(a.root), pathlib.Path(a.ra), pathlib.Path(a.rb)
    problems = []
    counts = {t: len(list((root / t / SIZE).glob("*.png"))) for t in
              ("data7", "data8", "data7_rr4", "data8_rr4")}
    out = {"round": "M43", "question": "16px 该发含模组训练池的 v7 还是现行 v8（采纳问题）",
           "meter": "pixfrac = 逐像素不同比例的材质均值（与 M39/M41/M42 逐字相同）",
           "anchors": {"D37": D37, "D38": D38, "D40": D40, "D_seed_M42": D_SEED},
           "anchor_caveat": "本轮是跨码本比较，与三个锚点不同类（预注册 (5)）",
           "prerun_prediction_P1": "SCREEN_GO",
           "counts": counts, "runs": {"A": str(ra), "B": str(rb)}}

    st = _stats(_load_pairs(root / "data7_rr4" / SIZE, root / "data8_rr4" / SIZE, problems))
    out["pair_rr4"] = st
    out["pair_raw_first"] = _stats(_load_pairs(root / "data7" / SIZE, root / "data8" / SIZE, []))

    checked = 0
    try:
        ca = json.load(open(ra / "config.json", encoding="utf-8"))
        cb = json.load(open(rb / "config.json", encoding="utf-8"))
        bad, checked = opcheck(counts, ca, cb, _md5(ra / "codebook.npy"), _md5(rb / "codebook.npy"),
                               _ckpt_step(ra), _ckpt_step(rb),
                               st["identical"] if st["n"] else None)
        out["cfg_diff"] = sorted(k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k))
        out["extra_file"] = {"A": ca.get("extra_file", "<MISSING>=train_extra.json(含模组)"),
                             "B": cb.get("extra_file", "<MISSING>")}
        out["domain"] = {"A": ca.get("domain"), "B": cb.get("domain")}
    except (FileNotFoundError, KeyError, OSError) as e:
        bad, checked = ["OPrun:missing(%s)" % e], checked + 1
    problems += bad
    out["n_checked"] = checked
    out["problems"] = problems

    if st["n"] == 0:
        out["screen"] = "VOID_NO_DATA"
        out["note"] = "已查 %d 项，_rr4 一对图都没量到 -> 什么都没测到（⛔ 这不是'没事'）" % checked
    else:
        out["screen"] = screen_of(st["pixfrac"])
        out["P1_correct"] = (out["screen"] == "SCREEN_GO")

    pathlib.Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print("== (M43) 补充训练数据量：操作检验 + 零 API 筛子 ==")
    print("  张数 %s" % json.dumps(counts))
    for k in ("pair_rr4", "pair_raw_first"):
        v = out[k]
        if v["n"] == 0:
            print("  %-14s 【禁】无数据（n=0）" % k)
        else:
            print("  %-14s n=%3d  pixfrac=%.4f  mae=%.4f  逐像素全同 %d/%d"
                  % (k, v["n"], v["pixfrac"], v["mae"], v["identical"], v["n"]))
    print("  锚点 D37=%.4f D38=%.4f D40=%.4f（(M42) 零对照 D_seed=%.4f，只登记）"
          % (D37, D38, D40, D_SEED))
    print("  已查 %d 项，problems=%d %s" % (checked, len(problems), problems[:5]))
    print("  筛子 = %s（跑前预测 SCREEN_GO）" % out["screen"])
    print("  已写 %s" % a.out)
    return 1 if problems else 0


def _meter():
    """筛子读数进判官阶段的报告；缺文件必须明说没查到。"""
    p = EXP / "m43_meter.json"
    if not p.exists():
        return {"status": "not_checked", "note": "m43_meter.json 不在 -> 这一项什么也没量到"}
    m = json.loads(p.read_text(encoding="utf-8"))
    return {"status": "ok", "screen": m.get("screen"),
            "pixfrac": (m.get("pair_rr4") or {}).get("pixfrac"),
            "mae": (m.get("pair_rr4") or {}).get("mae"),
            "identical": (m.get("pair_rr4") or {}).get("identical"),
            "P1_correct": m.get("P1_correct"), "problems": m.get("problems")}


def _secondary():
    """预注册 (6) 的次要项 + 预测 (P3)。⛔ 非判据。"""
    p = EXP / "m43_gen16.json"
    if not p.exists():
        return {"status": "not_checked", "note": "m43_gen16.json 不在"}
    rows = json.loads(p.read_text(encoding="utf-8"))
    thr = {"KID_x1e3": 4.656, "FID": 6.0, "FD_DINOv2": 20.0, "CLIP": 0.41}
    got, over_rr4 = {}, 0
    for pair in (("data7_rr4", "data8_rr4"), ("data7", "data8")):
        a, b = rows.get(pair[0]), rows.get(pair[1])
        if not a or not b:
            got["%s_vs_%s" % pair] = "not_checked"
            continue
        cells = {}
        for k, t in thr.items():
            if a.get(k) is None or b.get(k) is None:
                cells[k] = "not_checked"
                continue
            diff = abs(float(a[k]) - float(b[k]))
            cells[k] = {"a": a[k], "b": b[k], "diff": diff, "thr": t, "over": diff >= t}
            if pair[0].endswith("_rr4") and diff >= t:
                over_rr4 += 1
        cells["n_equal"] = (a.get("n") == b.get("n"))
        cells["materials_equal"] = (a.get("materials") == b.get("materials"))
        got["%s_vs_%s" % pair] = cells
    return {"status": "ok", "cells": got, "rr4_cells_over_thr": over_rr4,
            "P3_prediction": ">=1", "P3_correct": over_rr4 >= 1}


def run_judge():
    from judge_cluster import analyse
    raw_p = EXP / (ARM + ".json")
    if not raw_p.exists():
        print("【禁】判官 JSON 不存在: %s -> 已查 0 项，什么也没测到" % raw_p)
        return 2
    out, _decided, _recs = analyse(
        "M43_data16", ARM,
        "(M43) 16px TRD16c 配方：v7 step_20000（含模组池+domain）vs v8 step_20000（现行）")
    raw = json.loads(raw_p.read_text(encoding="utf-8"))

    rr, pf = raw.get("resolve_rate"), raw.get("p_vs_floor")
    resolvable = rr is not None and pf is not None and pf < 0.05 and rr > FLOOR
    g = out["grains"]["U1_drop_last"]
    lo, hi = g["ci"]
    verdict = decide(resolvable, out["verdict"], lo, hi)

    checks = {"OP1_recount_exact": out["OP1_recount_exact"],
              "OP2_degenerate_pass": out["OP2_degenerate"]["pass"],
              "OP3_pass": g["OP3_pass"], "OP4_boot_pass": g["OP4_pass"],
              "api_fail_zero": raw["api_fail"] == 0, "lofo_valid": g["lofo"]["valid"]}
    problems = [k for k, v in checks.items() if not v]

    res = {"arm": ARM, "verdict": verdict, "resolvable": resolvable,
           "resolve_rate": rr, "p_vs_floor": pf, "floor": FLOOR,
           "prerun_prediction_P2": "DATA_V7_WINS",
           "P2_correct": verdict == "DATA_V7_WINS",
           "binomial": {"a_wins": raw["a_wins"], "decided": raw["decided"], "rate": raw["rate"],
                        "p": raw["p"], "jeffreys": raw["jeffreys"],
                        "inconsistent": raw["inconsistent"], "api_fail": raw["api_fail"]},
           "family_U1": {"ci": [lo, hi], "n_families": g["n_families"],
                         "eff_families": g["eff_families"], "max_family": g["max_family"],
                         "width": g["width"], "widening": g["widening"],
                         "contains_half": g["contains_half"],
                         "lofo_n_folds": g["lofo"]["n_folds"],
                         "lofo_all_same_side": g["lofo"]["all_same_side"],
                         "lofo_flipped": len(g["lofo"]["flipped"])},
           "family_U2_report_only": {"ci": out["grains"]["U2_first"]["ci"],
                                     "contains_half": out["grains"]["U2_first"]["contains_half"]},
           "cluster_label": out["verdict"],
           "screen_report": _meter(),
           "secondary_report_only": _secondary(),
           "n_checked": len(checks), "problems": problems}
    OUT_JUDGE.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print("-> %s" % OUT_JUDGE)
    print("可解率 %.1f%% vs 地板 %.1f%% p=%.3g -> %s"
          % (rr * 100, FLOOR * 100, pf, "有分辨力" if resolvable else "【禁】不可解读"))
    print("裸二项（只当下界）: A(v7) 胜 %d/%d = %.1f%%  p=%.4g  Jeffreys [%.3f,%.3f]"
          % (raw["a_wins"], raw["decided"], raw["rate"] * 100, raw["p"],
             raw["jeffreys"][0], raw["jeffreys"][1]))
    print("族级 (U1) CI = [%.4f, %.4f]  族数 %d（有效 %.1f，最大族 %d）拓宽 %.2fx  含 0.5: %s"
          % (lo, hi, g["n_families"], g["eff_families"], g["max_family"],
             g["widening"], g["contains_half"]))
    print("LOFO %d fold，全同侧 %s，翻侧 %d  -> 聚类标签 %s"
          % (g["lofo"]["n_folds"], g["lofo"]["all_same_side"],
             len(g["lofo"]["flipped"]), out["verdict"]))
    s = res["screen_report"]
    if s["status"] == "ok":
        print("筛子（本轮跨码本、与锚点不同类）: %s pixfrac=%s" % (s["screen"], s["pixfrac"]))
    else:
        print("筛子 %s: %s" % (s["status"], s["note"]))
    print("操作检验：已查 %d 项，problems=%d %s" % (len(checks), len(problems), problems))
    print("判决 = %s（跑前预测 DATA_V7_WINS）" % verdict)
    return 0


# --------------------------------------------------------------------------- 自测
def selftest():
    import numpy as np
    ok, fail = 0, []

    def chk(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    z = np.zeros((16, 16, 3), np.uint8)
    o = np.full((16, 16, 3), 255, np.uint8)
    half = z.copy()
    half[:8] = 255
    chk("T1 同图 pixfrac=0", _stats([("a", z, z)])["pixfrac"] == 0.0)
    chk("T2 全异 pixfrac=1", _stats([("a", z, o)])["pixfrac"] == 1.0)
    chk("T3 半异 pixfrac=0.5", abs(_stats([("a", z, half)])["pixfrac"] - 0.5) < 1e-9)
    s = _stats([])
    chk("T4 空集报 n=0 而不是 0.0", s["n"] == 0 and s["pixfrac"] is None)
    chk("T5 GO", screen_of(0.90) == "SCREEN_GO")
    chk("T6 NOGO", screen_of(0.30) == "SCREEN_NOGO")
    chk("T7 之间", screen_of(0.70) == "SCREEN_AMBIGUOUS")
    chk("T8 边界 d==D37 算 GO", screen_of(D37) == "SCREEN_GO")
    chk("T9 边界 d==D38 算 NOGO", screen_of(D38) == "SCREEN_NOGO")
    chk("T10 D_seed 落在 AMBIGUOUS 带", screen_of(D_SEED) == "SCREEN_AMBIGUOUS")

    good = {"data7": 500, "data8": 500, "data7_rr4": 125, "data8_rr4": 125}
    ca = {"out": "a", "domain": True, "steps": 30000, "save_at": [1], "codebook_err": 7.9, "d": 384}
    cb = {"out": "b", "domain": False, "steps": 20000, "save_at": [2], "codebook_err": 7.2,
          "extra_file": "train_extra_packs_only.json", "d": 384}
    chk("T11 操作检验全过", opcheck(good, ca, cb, "m", "n", 20000, 20000, 3) == ([], 8))
    chk("T12 张数不对", any(x.startswith("OP1:") for x in
                            opcheck(dict(good, data7_rr4=124), ca, cb, "m", "n", 20000, 20000, 3)[0]))
    chk("T13 多出差异键", any(x.startswith("OP2:extra_keys") for x in
                              opcheck(good, ca, dict(cb, d=512), "m", "n", 20000, 20000, 3)[0]))
    chk("T14 该差的键没差", any(x.startswith("OP2:missing") for x in
                                opcheck(good, ca, dict(cb, domain=True), "m", "n", 20000, 20000, 3)[0]))
    chk("T15 步数不对", any(x.startswith("OP3:") for x in
                            opcheck(good, ca, cb, "m", "n", 20000, 30000, 3)[0]))
    chk("T16 码本相同=拿错 run", "OP4:codebook_md5_same" in
        opcheck(good, ca, cb, "m", "m", 20000, 20000, 3)[0])
    chk("T17 同图过多", "OP5:same=26" in opcheck(good, ca, cb, "m", "n", 20000, 20000, 26)[0])
    chk("T18 没量到图", "OP5:no_data" in opcheck(good, ca, cb, "m", "n", 20000, 20000, None)[0])

    cases = [((False, "W1_robust", 0.60, 0.80), "VOID_UNRESOLVABLE"),
             ((True, "VOID_no_data", 0.60, 0.80), "VOID_VOID_no_data"),
             ((True, "W1_robust", 0.55, 0.70), "DATA_V7_WINS"),
             ((True, "W1_robust", 0.22, 0.44), "DATA_V8_WINS"),
             ((True, "W2a", 0.45, 0.62), "DATA_NULL"),
             ((True, "W1_robust", 0.50, 0.70), "DATA_NULL"),
             ((True, "W1_robust", 0.30, 0.50), "DATA_NULL")]
    for args, want in cases:
        chk("T judge %s" % want, decide(*args) == want)

    for p in (EXP / (ARM + ".json"), EXP / "m43_meter.json", EXP / "m43_gen16.json"):
        print("[selftest] %-46s %s" % (p.name, "在" if p.exists() else "【禁】还没到，不能判"))
    print("[selftest] %d/%d 通过 %s" % (ok, ok + len(fail), fail))
    return 0 if not fail else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", choices=["screen", "judge"], default="judge")
    ap.add_argument("--root", default="/tmp/m43ab")
    ap.add_argument("--ra", default="runs/trd_v7")
    ap.add_argument("--rb", default="runs/trd_v8")
    ap.add_argument("--out", default="/tmp/m43_meter.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    return run_screen(a) if a.mode == "screen" else run_judge()


if __name__ == "__main__":
    sys.exit(main())
