"""(M33) 判读器：把 `scripts/m33_trd_floor.sh` 头部写死的判据 (1)(2)(3)(4) 机械执行一遍。

⚑ **本文件写于生成开跑之前**（/tmp/nf_arms 还是空的、noise_floor 的 JSON 不存在）
  —— 判读器对结果是盲的。⛔ 出结果后**一个字都不许改判据**。

为什么要有它（与 (M31) 同一个理由，且那一轮盲写期真抓到过三个 bug）：
  - 判据冻结了、读数却手抄；`2*sqrt(sd_a^2+sd_b^2)` 手算三个臂对极易抄错；
  - `bad` 天然为空时不许读成"通过"（拿"没量"冒充"量过没事"）-> 每条操作检验另报"已查几项"；
  - 落盘一律挪到打印之前（非 GBK 字形崩在写 JSON 之前，本项目已犯过三次）。

用法（在**远程**跑；纯标准库 + math，零 GPU、零第三方依赖）：
  python analysis/arch/m33_read_floor.py --floor /tmp/m33_noise_floor_Vmat_16.json \
      --out /tmp/m33_floor.json
自测（伪造数据，跑完即删；⛔ 自测产物不许入库）：
  python analysis/arch/m33_read_floor.py --selftest
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m31_read_drift import op1                                   # noqa: E402  判据①(OP1) 逐字复用

V8_FLOOR = {1: 4.76, 2: 4.45, 4: 2.70}   # v8 检索基线上的 KID_x1e3 门槛，⛔ 不许改
BAND = 0.10                              # 判据③的 +-10% 带宽，⛔ 不许放宽
K = 8                                    # 每材质张数
N_MAT = 125                              # V_mat 的材质数
N_DRAWS = 12                             # noise_floor.py 的默认抽样次数
METRIC = "KID_x1e3"                      # ⚠ 键名不是 "KID"（(M30) 硬规矩③）

# (名字, run 目录, 生成 tag)
ARMS = [
    ("seed0", ROOT / "runs/trd_v11d", "nfs0"),
    ("seed1", Path("/tmp/runs/trd_seed1_09161230"), "nfs1"),
    ("seed2", Path("/tmp/runs/trd_seed2_09161230"), "nfs2"),
]


def op2(root: Path):
    """每条臂 125 材质 x 8 张，且每材质齐整 8 张。返回 (通过?, 问题, 已查臂数)。"""
    bad, checked = [], []
    for name, _run, tag in ARMS:
        d = root / tag / "16"
        if not d.is_dir():
            bad.append(f"{name}: 缺目录 {d}")
            continue
        checked.append(name)
        pngs = list(d.glob("*.png"))
        per = {}
        for p in pngs:
            per[p.stem.rsplit("_", 1)[0]] = per.get(p.stem.rsplit("_", 1)[0], 0) + 1
        if len(pngs) != N_MAT * K:
            bad.append(f"{name}: {len(pngs)} png != {N_MAT * K}")
        if len(per) != N_MAT:
            bad.append(f"{name}: {len(per)} 个材质 != {N_MAT}")
        uneven = {s: c for s, c in per.items() if c != K}
        if uneven:
            bad.append(f"{name}: {len(uneven)} 个材质张数 != {K}（例 {list(uneven.items())[:3]}）")
    return (not bad and len(checked) == len(ARMS)), bad, checked


def op3(tbl):
    """JSON 里三条臂齐全、m 键含 {1,2,4,8}，m<8 时 draws == 12。返回 (通过?, 问题, 已查项数)。"""
    bad, checked = [], 0
    for name, _run, tag in ARMS:
        if tag not in tbl:
            bad.append(f"{name}: JSON 里没有 {tag}")
            continue
        for m in (1, 2, 4, K):
            if str(m) not in tbl[tag]:
                bad.append(f"{name}: 缺 m={m}")
                continue
            cell = tbl[tag][str(m)].get(METRIC)
            if cell is None:
                bad.append(f"{name} m={m}: 缺 {METRIC}")
                continue
            checked += 1
            if m < K and cell.get("draws") != N_DRAWS:
                bad.append(f"{name} m={m}: draws={cell.get('draws')} != {N_DRAWS}")
            if cell.get("sd") is None or cell.get("mean") is None:
                bad.append(f"{name} m={m}: sd/mean 缺")
    return (not bad and checked == len(ARMS) * 4), bad, checked


def thresholds(tbl, m):
    """返回 (每臂 sd, thr_max, thr_mean)。thr = 2*sqrt(sd_a^2 + sd_b^2)（项目判据原式）。"""
    sd = {n: tbl[tag][str(m)][METRIC]["sd"] for n, _r, tag in ARMS}
    names = [n for n, _r, _t in ARMS]
    pairwise = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pairwise[f"{a}-{b}"] = 2 * math.hypot(sd[a], sd[b])
    thr_mean = 2 * math.sqrt(2) * (sum(sd.values()) / len(sd))
    return sd, pairwise, max(pairwise.values()), thr_mean


def drifts(tbl, m):
    """三条臂 KID 均值两两绝对差。⚠ n=3，只报区间、⛔ 无 p 值。"""
    mean = {n: tbl[tag][str(m)][METRIC]["mean"] for n, _r, tag in ARMS}
    names = [n for n, _r, _t in ARMS]
    d = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            d[f"{a}-{b}"] = abs(mean[a] - mean[b])
    return mean, d, max(d.values())


def selftest():
    """伪造一份 noise_floor JSON，确认判读器不会崩、判决三个分支都能到达。"""
    import tempfile
    ok_all = True
    for label, sd1, expect in [("紧", 0.30, "FLOOR_OVERSTATED"),
                               ("齐", 4.76 / (2 * math.sqrt(2)), "FLOOR_AGREES"),
                               ("吵", 3.00, "FLOOR_UNDERSTATED")]:
        tbl = {tag: {str(m): {METRIC: {"mean": 6.0 + i * 0.5, "sd": sd1, "draws": N_DRAWS if m < K else 1,
                                       "min": 0.0, "max": 1.0}}
                     for m in (1, 2, 4, K)}
               for i, (_n, _r, tag) in enumerate(ARMS)}
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td) / "f.json"
            fp.write_text(json.dumps(tbl), encoding="utf-8")
            out = Path(td) / "o.json"
            code = run(fp, out, Path(td) / "nope", skip_ops=True)
            got = json.loads(out.read_text(encoding="utf-8")).get("verdict")
            hit = got == expect
            ok_all &= hit
            print(f"[自测 {label}] 期望 {expect} 实得 {got} -> {'OK' if hit else '不符'}（退出码 {code}）")
    # 缺键 / 空目录：必须判 OPS_FAILED 而不是崩，也不许把"没量"读成通过
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "empty.json"
        fp.write_text("{}", encoding="utf-8")
        out = Path(td) / "o2.json"
        run(fp, out, Path(td) / "nope", skip_ops=False)
        got = json.loads(out.read_text(encoding="utf-8"))
        hit = got.get("verdict") == "OPS_FAILED" and got["ops"]["op3_checked"] == 0
        ok_all &= hit
        print(f"[自测 空JSON] 期望 OPS_FAILED 且 op3_checked==0 -> {'OK' if hit else '不符'}")
    print("自测" + ("全过" if ok_all else "有不符项"))
    return 0 if ok_all else 1


def run(floor_path: Path, out_path: Path, root: Path, skip_ops: bool):
    res = {"v8_floor": V8_FLOOR, "band": BAND, "arms": [a[0] for a in ARMS],
           "floor_json": str(floor_path)}
    if not floor_path.exists():
        res["verdict"] = "PENDING"
        res["ops"] = {"note": f"{floor_path} 还不存在"}
        out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{floor_path} 还不存在 -> PENDING（【禁】这不叫通过）")
        return 3
    tbl = json.loads(floor_path.read_text(encoding="utf-8"))

    # ---- (1) 操作检验 ----
    if skip_ops:                                   # 只在自测里走这条路
        ok1, bad1, seeds, ok2, bad2, chk2 = True, [], {}, True, [], ["自测跳过"]
    else:
        cfgp = {n: r / "config.json" for n, r, _t in ARMS}
        miss = [f"{n}: 缺 {p}" for n, p in cfgp.items() if not p.exists()]
        if miss:
            ok1, bad1, seeds = False, miss, {}
        else:
            ok1, bad1, seeds = op1({n: json.loads(p.read_text(encoding="utf-8")) for n, p in cfgp.items()})
        ok2, bad2, chk2 = op2(root)
    ok3, bad3, chk3 = op3(tbl)
    res["ops"] = {"op1": ok1, "op1_diffs": bad1, "seeds": seeds,
                  "op2": ok2, "op2_problems": bad2, "op2_arms_checked": chk2,
                  "op3": ok3, "op3_problems": bad3, "op3_checked": chk3}

    if not (ok1 and ok2 and ok3):
        res["verdict"] = "OPS_FAILED"
        out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"(OP1) 配方逐键比对：{'通过' if ok1 else '不通过'}；seed = {seeds}")
        for b in bad1:
            print("  差异 " + b)
        print(f"(OP2) 每臂 {N_MAT}x{K} 张：{'通过' if ok2 else '不通过'}（已查臂 {chk2}）")
        for b in bad2:
            print("  " + b)
        print(f"(OP3) JSON 结构/draws：{'通过' if ok3 else '不通过'}（已查 {chk3} 项，"
              f"应为 {len(ARMS) * 4} 项）")
        for b in bad3:
            print("  " + b)
        print("\n=> OPS_FAILED，按预注册①不下判决。")
        return 2

    # ---- (2) 主统计量 + (3) 判决 ----
    per_m = {}
    for m in (1, 2, 4):
        sd, pw, thr_max, thr_mean = thresholds(tbl, m)
        mean, dd, dmax = drifts(tbl, m)
        per_m[m] = {"sd": sd, "thr_pairwise": pw, "thr_max": thr_max, "thr_mean": thr_mean,
                    "kid_mean": mean, "pairwise_abs_diff": dd, "Dmax": dmax,
                    "v8": V8_FLOOR[m], "ratio_vs_v8": thr_max / V8_FLOOR[m]}
    r = per_m[1]["ratio_vs_v8"]
    verdict = ("FLOOR_UNDERSTATED" if r > 1 + BAND else
               "FLOOR_OVERSTATED" if r < 1 - BAND else "FLOOR_AGREES")
    res["per_m"] = per_m
    res["ratio_m1"] = r
    res["verdict"] = verdict
    out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")   # 先落盘

    print(f"(OP1) 配方逐键比对：通过；seed = {seeds}")
    print(f"(OP2) 每臂 {N_MAT}x{K} 张：通过（已查臂 {chk2}）")
    print(f"(OP3) JSON 结构/draws：通过（已查 {chk3} 项）")
    print(f"\n== 判据② TRD 臂上的 {METRIC} 守门门槛（thr = 2*sqrt(sd_a^2+sd_b^2)）==")
    for m in (1, 2, 4):
        p = per_m[m]
        print(f"  m={m}  每臂 sd " + " ".join(f"{n}={v:.3f}" for n, v in p["sd"].items())
              + f"  | thr_max={p['thr_max']:.3f}（次要 thr_mean={p['thr_mean']:.3f}）"
              + f"  | v8={p['v8']:.2f}  比值={p['ratio_vs_v8']:.2f}")
    print(f"\n=> 判 {verdict}（m=1 比值 {r:.2f}，带宽 +-{BAND:.0%}）")
    if verdict == "FLOOR_UNDERSTATED":
        print(f"   今后守门下限改用 TRD 实测：m=1 -> {per_m[1]['thr_max']:.2f}，"
              f"m=4 -> {per_m[4]['thr_max']:.2f}。")
        print("   【禁】不重开任何已下判决：(M29) 的 +2.118 本就落在 4.8 内，下限变大只会更落在里面。")
    elif verdict == "FLOOR_OVERSTATED":
        print("   v8 的数偏保守，仍可继续引用（保守方向安全）；改用 TRD 数收紧门时，"
              "新臂预注册必须显式写明用的是哪一个数。")
        print("   【禁】不许回头用更松的门复审任何已判的臂（(M29)/(M31) 判决一个字不改）。")
    else:
        print("   4.76/2.70 沿用，并首次获得「在 TRD 臂上也成立」的凭据。")

    print(f"\n== 判据④ 重训漂移（报告项）==")
    for m in (1, 2, 4):
        p = per_m[m]
        print(f"  m={m}  每臂均值 " + " ".join(f"{n}={v:.3f}" for n, v in p["kid_mean"].items())
              + "  | 两两绝对差 " + " ".join(f"{n}={v:.3f}" for n, v in p["pairwise_abs_diff"].items())
              + f"  | Dmax={p['Dmax']:.3f}")
    print(f"  m=4 守门要画在 max(thr_max={per_m[4]['thr_max']:.2f}, Dmax={per_m[4]['Dmax']:.2f})"
          f" = {max(per_m[4]['thr_max'], per_m[4]['Dmax']):.2f} 之外；"
          "用 (M32) 成对配方可消掉 Dmax 那一项。")
    print("  【禁】本轮 Dmax(1) 不是 (M31) 的 1.087（估计量不同：这里是 12 次抽样的均值，"
          "(M31) 是单次固定取第 0 张）-> 不许说复现/未复现，也不许替换 (M31) 的读数。")
    print("  【禁】n=3 个差值，只报区间、无 p 值；三条臂只差 seed，不比优劣。")
    print(f"\n-> {out_path}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=Path, default=Path("/tmp/m33_noise_floor_Vmat_16.json"))
    ap.add_argument("--root", type=Path, default=Path("/tmp/nf_arms"))
    ap.add_argument("--out", type=Path, default=Path("/tmp/m33_floor.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    raise SystemExit(selftest() if a.selftest else run(a.floor, a.out, a.root, skip_ops=False))


if __name__ == "__main__":
    main()
