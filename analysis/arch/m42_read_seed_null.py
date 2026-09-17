#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(M42) 零 API 筛子的**构造性零对照**判读器 —— 盲写于任何一张图存在之前。

背景：(M41) 起 pixfrac 筛子被**当门用**（非 SCREEN_GO 就 exit 3、零 API），并且第一次真的
挡下了一小时 API。但它的锚点只有 n=3（D37 显著 / D38 NULL / D40 NULL），**从来没有独立验证**；
而门的固有代价是「它说 NOGO 的那次恰好也不会给它评分」⇒ 靠开臂永远攒不到验证。

本轮买的是**另一种**验证，零 API 就能买到：**构造性零对照**。
项目里唯一一对**只换随机种子**的重训（`/tmp/runs/trd_seed{1,2}_09161230`，config 只差
`out`/`seed`，同一份 codebook、同一个 `--init_from runs/trd_v10/last.pt`、同为 12000 步）
的**真效应按构造为 0**。用与 (M37)(M38)(M40)(M41) **逐字相同**的出图配方量它们之间的
D_seed = pixfrac，就知道「这把尺子在**跨 run 重训**这一类比较上，纯噪声能顶到多高」。

⚠ 这不回答"哪个种子更好"，⛔ 它的任何数字都不是质量证据，⛔ 不许写进任何主张。

判据（跑前写死，⛔ 跑完不许改、不许换尺子、不许挪锚点）
--------------------------------------------------------------------------------
(C0) 操作检验，任一不过 -> `VOID_OPCHECK`（⛔ 不许拿空集冒充"量过没事"）：
     - 两臂各 500 张（125 材质 x n=4）、各 125 张 `_rr4`
     - 两份 config 的差异键 ⊆ {out, seed, codebook_err}
     - 两份 codebook.npy 的 md5 相同
     - 两臂末步都是 12000
     ⚠ **逐像素相同的张数只登记、不作 VOID 触发器**：本轮"差别小"是合法读数而不是失败，
       拿它作废会把最有信息量的那个结果扔掉（M41 的 OP5 是另一回事：那里要排除"没跑起来"）。
(V1) D_seed >= D37 = 0.8173  -> `SEED_NULL_TRIVIAL`
     ＝ 筛子的 GO 线连**纯种子噪声**都挡不住。授权且仅授权一句话：
     ⛔ **今后不许把筛子当门用在"两条独立重训之间"的臂上**（成对臂——同 init、同种子、
     同数据序、只改一个变量，如 (M41)——不受影响，那一类的噪声零点是另一个数）。
(V2) D_seed <= D40 = 0.6431  -> `SEED_NULL_OK`
     ＝ 纯噪声低于最低锚点 ⇒ 筛子在跨 run 类上仍有余量，继续当门用，
     并把 D_seed 登记为**第四个锚点（零对照点）**。
(V3) 之间 -> `SEED_NULL_AMBIGUOUS` ＝ ⛔ 不改门、不挪锚点，只登记；
     今后引筛子必须同引这个读数。
(P1) **跑前预测：`SEED_NULL_TRIVIAL`**。理由（也写死，事后不许改口）：两臂都从同一个 v10
     出发各走 12000 步、方向互不相干 ⇒ 互相之间的距离应当不小于 (M40) 量到的"沿同一条轨迹
     再走 12000 步"（D40=0.6431），多半还更大。

⛔ 本轮零 API、不开判官臂、不改 D37/D38/D40、不改 `judge_pairs.py`、不改 `final_test.sh`。

用法：
  python analysis/arch/m42_read_seed_null.py --selftest
  python analysis/arch/m42_read_seed_null.py --root /tmp/m42seed \
      --ra /tmp/runs/trd_seed1_09161230 --rb /tmp/runs/trd_seed2_09161230 \
      --out /tmp/m42_seed_null.json
"""
import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

SIZE = "16"
D37, D38, D40 = 0.8173, 0.6608, 0.6431      # ⛔ 锚点写死，一个字不许改
WANT_STEP = 12000
CFG_ALLOWED = {"out", "seed", "codebook_err"}


def _stats(pairs):
    """pairs: [(name, arrA, arrB)] -> 逐像素差别统计（与 (M39)/(M41) 的尺子逐字相同）。"""
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
            "pixfrac": float(np.mean(fr)), "mae": float(np.mean(ae))}


def verdict(d_seed):
    """(V1)(V2)(V3) 三分判决。⚠ 闭区间边界：>=D37 算 TRIVIAL、<=D40 算 OK。"""
    if d_seed >= D37:
        return "SEED_NULL_TRIVIAL"
    if d_seed <= D40:
        return "SEED_NULL_OK"
    return "SEED_NULL_AMBIGUOUS"


def opcheck(counts, cfg_a, cfg_b, md5_a, md5_b, step_a, step_b):
    """(C0)。返回 (problems, checked)；problems 非空 -> VOID_OPCHECK。"""
    bad, checked = [], 0
    for tag, want in (("seed1", 500), ("seed2", 500), ("seed1_rr4", 125), ("seed2_rr4", 125)):
        checked += 1
        if counts.get(tag) != want:
            bad.append("OP1:%s=%s" % (tag, counts.get(tag)))
    checked += 1
    diff = sorted(k for k in set(cfg_a) | set(cfg_b) if cfg_a.get(k) != cfg_b.get(k))
    if set(diff) - CFG_ALLOWED:
        bad.append("OP2:cfg_diff=%s" % (sorted(set(diff) - CFG_ALLOWED),))
    if "seed" not in diff:
        bad.append("OP2:seed_not_different")      # 两臂必须**确实**换了种子
    checked += 1
    if md5_a != md5_b:
        bad.append("OP3:codebook_md5")
    checked += 1
    if not (step_a == step_b == WANT_STEP):
        bad.append("OP4:steps=%s/%s" % (step_a, step_b))
    return bad, checked


def _md5(p):
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


def _load_pairs(da, db, problems):
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


def selftest():
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
    s = _stats([("a", z, z)])
    chk("T1 同图 pixfrac=0", s["pixfrac"] == 0.0 and s["identical"] == 1)
    s = _stats([("a", z, o)])
    chk("T2 全异 pixfrac=1", s["pixfrac"] == 1.0 and abs(s["mae"] - 1.0) < 1e-9)
    s = _stats([("a", z, half)])
    chk("T3 半异 pixfrac=0.5", abs(s["pixfrac"] - 0.5) < 1e-9)
    s = _stats([])
    chk("T4 空集报 n=0 而不是 0.0", s["n"] == 0 and s["pixfrac"] is None)
    chk("T5 TRIVIAL", verdict(0.90) == "SEED_NULL_TRIVIAL")
    chk("T6 OK", verdict(0.30) == "SEED_NULL_OK")
    chk("T7 之间", verdict(0.70) == "SEED_NULL_AMBIGUOUS")
    chk("T8 边界 d==D37 算 TRIVIAL", verdict(D37) == "SEED_NULL_TRIVIAL")
    chk("T9 边界 d==D40 算 OK", verdict(D40) == "SEED_NULL_OK")
    good = {"seed1": 500, "seed2": 500, "seed1_rr4": 125, "seed2_rr4": 125}
    ca, cb = {"out": "a", "seed": 1, "d": 384}, {"out": "b", "seed": 2, "d": 384}
    bad, checked = opcheck(good, ca, cb, "m", "m", 12000, 12000)
    chk("T10 操作检验全过", bad == [] and checked == 7)
    chk("T11 张数不对", "OP1:seed1_rr4=124" in opcheck(
        dict(good, seed1_rr4=124), ca, cb, "m", "m", 12000, 12000)[0])
    chk("T12 config 多出差异键", any(x.startswith("OP2:cfg_diff") for x in opcheck(
        good, ca, dict(cb, d=512), "m", "m", 12000, 12000)[0]))
    chk("T13 种子没换", "OP2:seed_not_different" in opcheck(
        good, ca, dict(ca), "m", "m", 12000, 12000)[0])
    chk("T14 码本不同", "OP3:codebook_md5" in opcheck(good, ca, cb, "m", "n", 12000, 12000)[0])
    chk("T15 步数不对", any(x.startswith("OP4:") for x in opcheck(
        good, ca, cb, "m", "m", 12000, 11000)[0]))
    chk("T16 全同图不作废", opcheck(good, ca, cb, "m", "m", 12000, 12000)[0] == [])
    print("[selftest] %d/%d 通过 %s" % (ok, ok + len(fail), fail))
    return 0 if not fail else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/m42seed")
    ap.add_argument("--ra", default="/tmp/runs/trd_seed1_09161230")
    ap.add_argument("--rb", default="/tmp/runs/trd_seed2_09161230")
    ap.add_argument("--out", default="/tmp/m42_seed_null.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())

    root, ra, rb = pathlib.Path(a.root), pathlib.Path(a.ra), pathlib.Path(a.rb)
    problems = []
    counts = {t: len(list((root / t / SIZE).glob("*.png"))) for t in
              ("seed1", "seed2", "seed1_rr4", "seed2_rr4")}
    out = {"round": "M42", "question": "pixfrac 筛子在『跨 run 重训』上的构造性零对照",
           "meter": "pixfrac = 逐像素不同比例的材质均值（与 M39/M41 逐字相同）",
           "anchors": {"D37": D37, "D38": D38, "D40": D40},
           "prerun_prediction_P1": "SEED_NULL_TRIVIAL",
           "counts": counts, "runs": [str(ra), str(rb)]}
    checked = 0
    try:
        ca = json.load(open(ra / "config.json", encoding="utf-8"))
        cb = json.load(open(rb / "config.json", encoding="utf-8"))
        la = json.load(open(ra / "log.json", encoding="utf-8"))
        lb = json.load(open(rb / "log.json", encoding="utf-8"))
        bad, checked = opcheck(counts, ca, cb, _md5(ra / "codebook.npy"), _md5(rb / "codebook.npy"),
                               la[-1]["step"], lb[-1]["step"])
        out["cfg_diff"] = sorted(k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k))
        out["last_step"] = {"A": la[-1]["step"], "B": lb[-1]["step"]}
        out["val_last"] = {"A": la[-1]["val"], "B": lb[-1]["val"]}   # ⛔ val 不是判据（(M37)）
    except (FileNotFoundError, KeyError, IndexError) as e:
        bad, checked = ["OPtrain:missing(%s)" % e], checked + 1
    problems += bad

    st = _stats(_load_pairs(root / "seed1_rr4" / SIZE, root / "seed2_rr4" / SIZE, problems))
    out["pair_rr4"] = st
    # 参考：不经 4 选 1 的第 0 张（只登记，判决只用 _rr4 —— 三个锚点都是 _rr4 上量的）
    out["pair_raw_first"] = _stats(_load_pairs(root / "seed1" / SIZE, root / "seed2" / SIZE, []))
    out["n_checked"] = checked + 1
    out["problems"] = problems

    if st["n"] == 0:
        out["verdict"] = "VOID_NO_DATA"
        out["note"] = "已查 %d 项，_rr4 一对图都没量到 -> 本轮什么都没测到（⛔ 这不是'没事'）" % out["n_checked"]
    elif problems:
        out["verdict"] = "VOID_OPCHECK"
        out["note"] = "已查 %d 项，%d 项不过 -> 按 (C0) 本轮作废" % (out["n_checked"], len(problems))
    else:
        d = st["pixfrac"]
        out["D_seed"] = d
        out["verdict"] = verdict(d)
        out["P1_correct"] = (out["verdict"] == out["prerun_prediction_P1"])

    pathlib.Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    # ⚠ 落盘已在打印之前完成（非 GBK 字形会让 Windows 控制台崩在写文件之前）
    print("== (M42) 筛子的构造性零对照（零 API）==")
    print("  张数 %s" % json.dumps(counts))
    for k in ("pair_rr4", "pair_raw_first"):
        v = out[k]
        if v["n"] == 0:
            print("  %-14s 【禁】无数据（n=0）" % k)
        else:
            print("  %-14s n=%3d  pixfrac=%.4f  mae=%.4f  逐像素全同 %d/%d"
                  % (k, v["n"], v["pixfrac"], v["mae"], v["identical"], v["n"]))
    print("  锚点 D37=%.4f D38=%.4f D40=%.4f" % (D37, D38, D40))
    print("  已查 %d 项，problems=%d %s" % (out["n_checked"], len(problems), problems[:5]))
    print("  判决: %s（跑前预测 %s）" % (out["verdict"], out["prerun_prediction_P1"]))
    print("  已写 %s" % a.out)


if __name__ == "__main__":
    main()
