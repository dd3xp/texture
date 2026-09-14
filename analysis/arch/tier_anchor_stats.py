"""判读 `eval/tier_anchor.sh`：三档的 24->32 下滑是 TRD 掉下去，还是降采样族基线升上来？

零 GPU、零 API：只读 `eval/tier_anchor.sh` 落盘的四份判定 JSON，外加**已封盘**的 B2 那两份
（后者由 `recheck_judge.py` 每轮复核）。输出写本机 `experiments/tier_anchor.json`。

判据、方向预测、口径限制全部写在 `eval/tier_anchor.sh` 的文件头，**跑之前就提交了**（8ec5446）。
本脚本只是把那里的判据 (2) 机械化，**不新增任何判据**。摘录：

  (2) 主判据 = 梯度，不是胜率本身。对每个基线，在它 24 与 32 两条臂**共同判出**的材质上做
      McNemar（口径与 `tier_overlap.py` 逐字相同）。
      ⚠ 子集口径差一处，别把数字记混：`tier_overlap.py` 是在**三档共同**判出的 118 个材质上做的
      （18:5，p=0.011），本脚本每个基线只有两条臂，所以用的是**该基线 24&32 共同**判出的集合。
      B2 在后一个口径上是 n=150、47.3% -> 35.3%、翻转 **23:5**、**p=9.1e-04**。两者不矛盾，
      是同一件事的两个分母；本脚本对 B1/B7 用的就是后一个口径，三者可比。
  (3) 方向预测：预测**对 B1 下滑、对 B7 不下滑**，即 (H_B2)。
  (5) B7 在 24/32 从未调过 CFG -> 对 B7 的**绝对胜率照实报但不进判据**，只判梯度
      （同一个让步在 24 与 32 上完全相同，造不出梯度）。
  (6) B7 与 TRD 同一个训练池 -> "对 B7 不下滑"与 (H_B2) 相容但**不等于证明**；B1 才是不受这个
      弱点影响的那条腿。结论必须 B1 与 B7 合读。
  (7) 本轮不授权任何配置变更。

B2 那一对在这里**只作标尺**（它已经封盘，数字不会变）：它告诉我们这台仪器在这批材质上能把
多大的梯度判出来，从而让"对 B7 没判到梯度"这句话有一个可比的参照，而不是空口说"没效应"。

    python analysis/arch/tier_anchor_stats.py
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EXP = os.path.join(ROOT, "experiments")

# 基线 -> (24 那条臂的文件, 32 那条臂的文件, 是否进判据)
ARMS = {
    "B7": ("judge_full_TRD24_rr4_vs_B7_24.json", "judge_full_TRD32_rr4_vs_B7_32.json", True),
    "B1": ("judge_full_TRD24_rr4_vs_B1_24.json", "judge_full_TRD32_rr4_vs_B1_32.json", True),
    # 已封盘，只作标尺：tier_overlap.py 在三档共同子集上得到 24 vs 32 翻转 18:5、p=0.011
    "B2": ("judge_full_TRD24_rr4_vs_B2_24.json", "judge_full_TRD32_rr4_vs_B2_32.json", False),
}
LEDGER_B2 = {"24": (94, 188), "32": (83, 202)}   # 防读错文件


def binom_p(k, n):
    """Exact two-sided binomial test against p=0.5 (no scipy on the servers)."""
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) for i in range(n + 1)]
    obs = probs[k]
    return min(1.0, sum(p for p in probs if p <= obs + 1e-9) / float(2 ** n))


def load(fn):
    p = os.path.join(EXP, fn)
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def main():
    out = {"arms": {}, "gradient": {}}
    missing = []
    for base, (f24, f32, judged) in ARMS.items():
        d24, d32 = load(f24), load(f32)
        if d24 is None or d32 is None:
            missing.append(base)
            continue
        if base == "B2":       # O1: 标尺必须与账本对得上
            for t, d in (("24", d24), ("32", d32)):
                assert (d["a_wins"], d["decided"]) == LEDGER_B2[t], (t, d["a_wins"], d["decided"])
        # O2（判据 (8)）：可比对必须是 272 对，API 失败必须为 0，且两档逐对同序同材质
        r24, r32 = d24["records"], d32["records"]
        assert len(r24) == len(r32) == 272, (base, len(r24), len(r32))
        assert d24["api_fail"] == d32["api_fail"] == 0, (base, d24["api_fail"], d32["api_fail"])
        for i, (a, b) in enumerate(zip(r24, r32)):
            assert a["pair"] == b["pair"] == i and a["material"] == b["material"], (base, i)

        v = {"24": [r["verdict"] for r in r24], "32": [r["verdict"] for r in r32]}
        dec = {t: set(i for i, x in enumerate(v[t]) if x in ("A", "B")) for t in v}
        for t in ("24", "32"):
            k = sum(1 for i in dec[t] if v[t][i] == "A")
            out["arms"][f"{base}@{t}"] = {
                "decided": len(dec[t]), "resolvable_rate": round(len(dec[t]) / 272, 3),
                "trd_wins": k, "rate": round(k / len(dec[t]), 3) if dec[t] else None,
                "p": binom_p(k, len(dec[t])),
                # 判据 (5)：对 B7 的绝对胜率只报不判（24/32 上 CFG 从未调过）
                "level_enters_verdict": base != "B7"}

        C = sorted(dec["24"] & dec["32"])
        b = sum(1 for i in C if v["24"][i] == "A" and v["32"][i] == "B")   # 24 赢、32 输
        c = sum(1 for i in C if v["24"][i] == "B" and v["32"][i] == "A")
        p = binom_p(b, b + c)
        out["gradient"][base] = {
            "common_decided": len(C),
            "rate24_on_common": round(sum(1 for i in C if v["24"][i] == "A") / len(C), 3) if C else None,
            "rate32_on_common": round(sum(1 for i in C if v["32"][i] == "A") / len(C), 3) if C else None,
            "flip_24win_32lose": b, "flip_24lose_32win": c, "discordant": b + c, "p": p,
            "falls": bool(b > c and p < 0.05),
            "judged": judged,
            "mats_24win_32lose": [r24[i]["material"] for i in C
                                  if v["24"][i] == "A" and v["32"][i] == "B"]}

    if missing:
        print("缺这些基线的判定 JSON（臂还没跑完或没过试点门）：", ", ".join(missing))

    print("== 各臂（TRD 胜率；对 B7 的水平不进判据，见判据 (5)）")
    for k, a in out["arms"].items():
        flag = "" if a["level_enters_verdict"] else "   [只报不判]"
        print("  TRD vs %-8s  %3d/%3d = %5.1f%%  p=%.3g  判出率 %.0f%%%s"
              % (k, a["trd_wins"], a["decided"], 100 * a["rate"], a["p"],
                 100 * a["resolvable_rate"], flag))

    print("== 24->32 梯度（共同判出子集上的 McNemar，判据 (2)）")
    for base, g in out["gradient"].items():
        tag = "标尺（已封盘）" if not g["judged"] else "进判据"
        print("  vs %-3s  n=%3d  %5.1f%% -> %5.1f%%   翻转 %d:%d（不一致 %d）  p=%.3g  下滑=%s  [%s]"
              % (base, g["common_decided"], 100 * g["rate24_on_common"],
                 100 * g["rate32_on_common"], g["flip_24win_32lose"], g["flip_24lose_32win"],
                 g["discordant"], g["p"], g["falls"], tag))

    # ---- 判据 (4)：四种读法，跑之前就写死在 eval/tier_anchor.sh 里，这里只做查表
    if all(b in out["gradient"] for b in ("B1", "B7")):
        f1, f7 = out["gradient"]["B1"]["falls"], out["gradient"]["B7"]["falls"]
        verdict = {
            (True, False): "(H_B2)：降采样族随画布拿到免费增益。"
                           "『32px 缺口』要改写，账本的准入条件作废重写。仍不改任何配置（判据 (7)）。",
            (True, True): "(H_TRD)：TRD 确实随画布变差。准入条件原样保留，缺口在我们这边。",
            (False, False): "两个都不下滑 -> B2 在三个基线里是特例；两个假设都不成立。"
                            "记录后停，**不许事后再编第三个故事**（判据 (4)）。",
            (False, True): "对 B7 下滑但对 B1 不 -> 方向预测被推翻，"
                           "且『降采样族』作为一个族的说法同时死掉。",
        }[(f1, f7)]
        out["verdict"] = verdict
        print("== 判读（判据 (4) 查表，非事后解释）\n  " + verdict)
        print("  ⚠ 判据 (6)：B7 与 TRD 同池训练，『对 B7 不下滑』与 (H_B2) 相容但不是证明；"
              "结论由 B1 与 B7 合读。")

    os.makedirs(EXP, exist_ok=True)
    with open(os.path.join(EXP, "tier_anchor.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("-> experiments/tier_anchor.json")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
