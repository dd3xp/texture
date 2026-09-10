"""把第 75 轮那把「可解率」尺子**回头量一遍已经发表的每一条判官臂**。

## 为什么这一轮不用花一次 API 调用

第 75 轮（`2ea1f6e` / `985ef15` / `18f473a`）立了条规矩：新开判官臂之前，
先在 10-15 对上只量正反两序的一致率、不看胜负；低于预注册的可解率就不开。
那条规矩当天就把「3.2 对 4.5」那条臂否掉了（可解 53%，空对照 60%）。

但那把尺子只用在了**将来**的臂上。今天清点数据时发现：
`crop_scale_study.py` 那套协议**从第一天起就是两序各问一次、不一致记为
`inconsistent` 落盘**（见该文件 `p1 if p1 == p2 else "inconsistent"`）。
也就是说，**过去每一条判官臂的可解率一直躺在仓库里，没有人看过一眼**。
本脚本只读已入库 JSON：不导 torch、不联网、不生成任何图。

## 一致率与「可解」的关系，以及为什么 50% 是硬地板

设一对里判官真有内容偏好的概率为 r（可解），否则两次作答互相独立地乱猜。
则两序一致率 c = r + (1 - r)/2，于是 **r = 2c - 1**。

50% 这条地板不是估出来的，是**协议本身推出来的**。协议问两次：
第一次给 `[B, A]`、答"第一张"记作 before；第二次给 `[A, B]`、答"第一张"记作 after。
若判官完全不看内容、只以概率 q 选第一张，则"两次都记成 before"要求
第一次答第一张、第二次答第二张，概率 q(1-q)；"两次都记成 after"同理。
一致率 = 2q(1-q) <= 0.5，**在 q = 0.5 时取到最大值 0.5**。
所以：**一个不看内容的判官，其一致率不可能超过 50%**——
这是上界，不是平均值。一致率明显低于 50% 则说明存在位置偏好（q 偏离 0.5），
而不只是噪声。第 75 轮那个空对照实测 3/5 = 60%（n=5，Jeffreys 跨度极宽）
与这条推导相容，本脚本用推导出来的 0.50，不用那个点估计。

⚠ 反过来不成立：一致率高**不等于判得对**。判官可以稳定地偏爱错的那张
（B15 里 gpt-5.6-sol 就是逐条倒转还很自洽）。本脚本量的是**分辨力**，
不是正确性。

## 预注册（写于运行之前，本提交即预注册；事后不改）

**样本框不由我挑**：扫描 `experiments/*.json` 中所有"至少有一条记录的某个
字段取值为 `inconsistent`"的字段，每个 (文件, 字段) 就是一条臂。
发现到的文件集合与下面 `PINNED` 逐一比对，**对不上就退出**——
免得将来多出或少掉文件时口径悄悄变了。

**每条臂算三个数**：
  - 已答 = 该字段存在的记录数。缺字段的记录分两类，**都只作描述、不参与判读**：
    门没触发所以**根本没问**（`crop_*` 那套里 `fired == False` 的记录），
    与**问了但 API 调用失败**（`fired == True` 却没有 `vlm`，协议里 fired 必问）。
    ⚠ 这两类必须分开数：混在一起会把"没问"说成"失败"，
    把失败率从 4% 吹成 51%（本轮初稿就是这么错的，判据未受影响）；
  - 一致率 c = 可解 / (可解 + inconsistent)，Jeffreys 95% 区间；
  - 该臂在**可解子集**上偏离 50/50 的双侧精确二项 p。
    ⚠ 这个 p 是**筛子，不是已发表统计**：它把所有尺寸/分层混在一起、
    方向取记录里出现最多的两个值，只用来把臂分成"出过结论"和"没出结论"两类。
    已发表数字一律以各自脚本为准，本脚本一个字都不改。

**判据表**（c 的 Jeffreys 下界记作 lo）：

| 条件 | 结论 | 记号 |
| --- | --- | --- |
| 已答 < 10 | 估不出来，本轮不判 | REPORT-ONLY |
| lo > 0.50 | 该臂的判官确有分辨力 | ABOVE，正文不动 |
| lo <= 0.50 且筛子 p < 0.05 | **拿地板上的仪器出了显著结论** | FLAG-A |
| lo <= 0.50 且筛子 p >= 0.05 | 零结果与"仪器不可用"分不开 | FLAG-B |

**预注册的后果**（跑前写死，免得看到结果再挑说法）：
  - FLAG-A：该臂支撑的主张必须在正文里注明"该比较上判官的一致率不高于
    掷硬币上界"，若该主张是承重的，还要改写成未经该仪器验证；
  - FLAG-B：任何"此路已关"的读法必须改写成"该仪器分辨不了"——
    **零结果不等于没效应**；
  - ABOVE：不动。
本脚本**只出诊断**，正文改动另开提交。

**验证钩子（可以推翻整个审计的那一条）**：`posbias_*.json` 三份里的
`kind == "check"` 是**清晰对重糊**的注意力检查，差别一眼可辨，
按上面的模型应当接近 100% 一致。若三判官合并的 check 一致率 < 80%，
说明"一致率≈可解率"这套读法在本项目的判官上根本不成立，
本轮降级为纯描述、**不落任何 FLAG**（退出码 3）。
（check 与真题的比较对象不同，这条钩子验的是**协议**能不能在
"差别明显"时给出高一致率，不是给某个判官发通行证。）

退出码：0 = 无 FLAG；1 = 有 FLAG；2 = 扫描范围与 PINNED 对不上；3 = 验证钩子未过。

    python analysis/paired/judge_resolvable_audit.py
    python analysis/paired/judge_resolvable_audit.py --selftest
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exact import binom_test, jeffreys                               # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
FLOOR = 0.50
MIN_ANSWERED = 10
CHECK_HOOK = 0.80

# 跑前钉死的扫描范围。发现集合与它不一致即退出 2。
PINNED = sorted([
    "bestof_gained.json", "bestof_rule.json", "bestof_units.json",
    "crop_ctrl.json", "crop_fewunits.json", "crop_holdout60.json",
    "crop_render384.json", "crop_render512.json",
    "crop_res5.json", "crop_res5b.json", "crop_robust.json",
    "crop_scale_study.json", "crop_scale_study_run1.json",
    "point_sample_iso.json", "render_small_iso.json",
    "seam_crop.json", "seam_crop60.json",
    "spread_recognise.json", "spread_recognise53.json", "spread_rescale.json",
])


def discover(files):
    """找出所有 (文件, 字段) 臂：字段至少有一条记录取值为 inconsistent。"""
    arms = []
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, list) or not d or not isinstance(d[0], dict):
            continue
        keys = sorted(set().union(*[set(r.keys()) for r in d]))
        for k in keys:
            if any(r.get(k) == "inconsistent" for r in d):
                arms.append((f, k))
    return arms


def score(recs, field):
    """一条臂的可解/不可解/未答，以及可解子集上的筛子二项检验。"""
    vals = [r.get(field) for r in recs]
    answered = [v for v in vals if isinstance(v, str)]
    inc = sum(1 for v in answered if v == "inconsistent")
    resolved = [v for v in answered if v != "inconsistent"]
    top = Counter(resolved).most_common(2)
    if len(top) == 2:
        k, n = top[0][1], top[0][1] + top[1][1]
    else:                      # 全落在一个值上（含 0 个），退化成单侧极端
        k = n = top[0][1] if top else 0
    # 缺字段的两类：门没触发（没问）与 fired 却没答（API 失败）。
    # 只有带 `fired` 字段的那套协议能区分；其余记为 unknown。
    nogate = failed = unknown = 0
    for r in recs:
        if isinstance(r.get(field), str):
            continue
        if "fired" in r:
            if r["fired"]:
                failed += 1
            else:
                nogate += 1
        else:
            unknown += 1
    return {"answered": len(answered), "resolved": len(resolved), "inc": inc,
            "nogate": nogate, "failed": failed, "unknown": unknown,
            "k": k, "n": n,
            "p": binom_test(k, n) if n else float("nan"),
            "labels": [t[0] for t in top]}


def verdict(s):
    if s["answered"] < MIN_ANSWERED:
        return "REPORT-ONLY", (float("nan"), float("nan"))
    lo, hi = jeffreys(s["resolved"], s["answered"])
    if lo > FLOOR:
        return "ABOVE", (lo, hi)
    if s["n"] and s["p"] < 0.05:
        return "FLAG-A", (lo, hi)
    return "FLAG-B", (lo, hi)


def check_hook():
    """三份 posbias 的 kind=check 合并一致率——差别一眼可辨时协议该给高一致率。"""
    ok = tot = 0
    per = []
    for f in sorted((EXP / "annotate").glob("posbias_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        c = [r for r in d if r.get("kind") == "check"]
        a = sum(1 for r in c if r["pick1"] == r["pick2"])
        per.append((f.name.replace("posbias_", "").replace(".json", ""), a, len(c)))
        ok += a
        tot += len(c)
    return ok, tot, per


def selftest():
    """判据分支 + 那条 2q(1-q) 推导的数值检查。"""
    import random
    fails = []

    # 1) 不看内容的判官：模拟协议本身，一致率必须 <= 0.5，q=0.5 时取到 0.5。
    for q in (0.5, 0.2, 0.8):
        rng = random.Random(7)
        agree = 0
        N = 40000
        for _ in range(N):
            o1 = rng.random() < q          # 第一次答"第一张"
            o2 = rng.random() < q
            p1 = "before" if o1 else "after"
            p2 = "after" if o2 else "before"
            agree += (p1 == p2)
        got, want = agree / N, 2 * q * (1 - q)
        if abs(got - want) > 0.01:
            fails.append(f"地板推导 q={q}: 实测 {got:.3f} 期望 {want:.3f}")
        if got > 0.51:
            fails.append(f"地板被突破 q={q}: {got:.3f}")

    # 2) score 的计数
    recs = [{"vlm": "after"}] * 8 + [{"vlm": "before"}] * 2 + \
           [{"vlm": "inconsistent"}] * 5 + [{}] * 3
    s = score(recs, "vlm")
    if (s["answered"], s["resolved"], s["inc"], s["unknown"]) != (15, 10, 5, 3):
        fails.append(f"score 计数错: {s}")
    if (s["k"], s["n"]) != (8, 10):
        fails.append(f"score 二项口径错: {s['k']}/{s['n']}")

    # 门没触发（没问）不许被记成 API 失败——本轮初稿正是在这里把两类混掉了。
    s2 = score([{"fired": False}] * 7 + [{"fired": True}] * 2
               + [{"fired": True, "vlm": "after"}] * 3, "vlm")
    if (s2["nogate"], s2["failed"], s2["answered"]) != (7, 2, 3):
        fails.append(f"未答分类错: nogate={s2['nogate']} failed={s2['failed']}")

    # 3) 判据三分支各命中一次
    cases = [
        ([{"vlm": "after"}] * 5 + [{"vlm": "inconsistent"}] * 2, "REPORT-ONLY"),
        ([{"vlm": "after"}] * 40 + [{"vlm": "inconsistent"}] * 4, "ABOVE"),
        # 一致率贴地板、可解子集却显著 -> FLAG-A
        ([{"vlm": "after"}] * 18 + [{"vlm": "before"}] * 2
         + [{"vlm": "inconsistent"}] * 20, "FLAG-A"),
        # 一致率贴地板、可解子集不显著 -> FLAG-B
        ([{"vlm": "after"}] * 11 + [{"vlm": "before"}] * 9
         + [{"vlm": "inconsistent"}] * 20, "FLAG-B"),
    ]
    for recs, want in cases:
        got, _ = verdict(score(recs, "vlm"))
        if got != want:
            fails.append(f"判据分支错: 期望 {want} 得到 {got}")

    # 4) 空字段不该被当成一条臂
    tmp = [{"vlm": "after"}, {"other": "inconsistent"}]
    ks = [k for k in ("vlm", "other") if any(r.get(k) == "inconsistent" for r in tmp)]
    if ks != ["other"]:
        fails.append(f"臂发现错: {ks}")

    for m in fails:
        print("  FAIL", m)
    print(f"selftest: {'全过' if not fails else str(len(fails)) + ' 处失败'}")
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())

    files = sorted(EXP.glob("*.json"))
    arms = discover(files)
    found = sorted({f.name for f, _ in arms})
    if found != PINNED:
        print("扫描范围与预注册的 PINNED 不一致，拒绝判读。")
        print("  多出:", [x for x in found if x not in PINNED])
        print("  缺少:", [x for x in PINNED if x not in found])
        raise SystemExit(2)

    ok, tot, per = check_hook()
    rate = ok / tot if tot else 0.0
    print("验证钩子（posbias 的注意力检查，清晰对重糊，差别一眼可辨）")
    for name, k, n in per:
        print(f"  {name:<28} {k}/{n}")
    print(f"  合并 {ok}/{tot} = {rate:.0%}   （预注册门槛 {CHECK_HOOK:.0%}）")
    hook_ok = rate >= CHECK_HOOK
    print(f"  -> {'通过，可按判据落 FLAG' if hook_ok else '未过，本轮降级为纯描述'}\n")

    print(f"{'臂':<34}{'已答':>5}{'可解':>5}{'一致率':>8}"
          f"{'Jeffreys 95%':>16}{'筛子 p':>10}  判读")
    rows = []
    for f, k in sorted(arms, key=lambda t: (t[0].name, t[1])):
        d = json.loads(f.read_text(encoding="utf-8"))
        s = score(d, k)
        v, (lo, hi) = verdict(s)
        name = f.name.replace(".json", "") + ("" if k == "vlm" else f":{k}")
        c = s["resolved"] / s["answered"] if s["answered"] else float("nan")
        ci = "-" if lo != lo else f"[{lo:.0%},{hi:.0%}]"
        print(f"{name:<34}{s['answered']:>5}{s['resolved']:>5}{c:>8.0%}"
              f"{ci:>16}{s['p']:>10.3g}  {v}")
        rows.append((name, s, v, lo))

    ans = sum(r[1]["answered"] for r in rows)
    res = sum(r[1]["resolved"] for r in rows)
    nog = sum(r[1]["nogate"] for r in rows)
    fai = sum(r[1]["failed"] for r in rows)
    unk = sum(r[1]["unknown"] for r in rows)
    print(f"\n合计 已答 {ans}   可解 {res} = {res/ans:.0%}")
    print(f"  未答的分两类（都不参与判读）：门没触发所以没问 {nog} 例；"
          f"问了但 API 失败 {fai} 例 = 已问的 {fai/max(fai+ans-unk,1):.0%}；"
          f"另有 {unk} 例所在协议无 fired 字段，分不出是哪类")
    print(f"按 r = 2c - 1，整个判官后目录的加权可解比例约 {max(0.0, 2*res/ans-1):.0%}")

    # --- 以下为描述性附录，写于看到结果之后，不参与任何判读 ---
    # 弃样是按"两序不一致"选的，不是按胜负选的，但选择毕竟发生了。
    # 最保守的界：把该臂弃掉的对**全部**算给劣势一方 / 全部算给优势一方，
    # 看已发表的方向还在不在。正文对 B13 已经做过"三种弃样处理"，这里补齐其余臂。
    # 单独看 FLAG 臂会得出"全都翻"这个同义反复（弃样一多，最坏界必然吞掉效应）。
    # 有信息的是**对照**：同一个界在 ABOVE 臂上翻不翻。翻不翻由弃样比例决定，
    # 而弃样比例正是一致率——这一栏因此是判据的独立佐证，不是判据本身。
    print("\n弃样敏感性（描述性，事后加的，不参与判读）：把弃掉的对**全部**算给")
    print("劣势一方，看方向与显著性还在不在。弃样一多，最坏界必然失效。")
    print(f"{'臂':<32}{'判读':>12}{'可解子集':>14}{'最坏界':>22}  结果")
    survive = {"ABOVE": [0, 0], "FLAG": [0, 0]}
    for name, s, v, lo in rows:
        if v == "REPORT-ONLY" or not s["n"]:
            continue
        k, n, m = s["k"], s["n"], s["inc"]
        wk, wn = k, n + m
        pw = binom_test(wk, wn)
        held = (wk / wn > 0.5) and pw < 0.05
        grp = "ABOVE" if v == "ABOVE" else "FLAG"
        survive[grp][1] += 1
        survive[grp][0] += held
        print(f"{name:<32}{v:>12}{f'{k}/{n}={k/n:.0%}':>14}"
              f"{f'{wk}/{wn}={wk/wn:.0%} p={pw:.3g}':>22}"
              f"  {'方向与显著性都在' if held else '吞掉了'}")
    for g in ("ABOVE", "FLAG"):
        h, t = survive[g]
        print(f"  {g:<6} 最坏界下仍成立 {h}/{t}")

    flags = [r for r in rows if r[2].startswith("FLAG")]
    if not hook_ok:
        print("\n验证钩子未过 -> 上面只作描述，不落 FLAG。")
        raise SystemExit(3)
    if not flags:
        print("\n无 FLAG：每条臂的一致率下界都高于掷硬币上界，正文不需改动。")
        raise SystemExit(0)
    print(f"\n{len(flags)} 条臂落在地板上（按预注册的后果处理）：")
    for name, s, v, lo in flags:
        why = ("拿地板上的仪器出了显著结论" if v == "FLAG-A"
               else "零结果与仪器不可用分不开")
        print(f"  [{v}] {name}: 一致率下界 {lo:.0%} <= {FLOOR:.0%} —— {why}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
