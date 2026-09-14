"""把「三档」和「B2 画布增益」放到同一把潜在尺子上，算出第四条边的**点预测**。

零 API、零 GPU：全部从已发表的判定 JSON 的逐对记录重算。

================================================================ 为什么可以这么算
`judge_full_B2_vs_B2up16_32.json`（预注册 `557a50e`）量到 **B2@32 打赢 B2@16 = 107/138 = 78%**。
加上已发表的两档，四个格子之间有**四条边、构成一个环**：

        TRD@16  --e1(57%)-->  B2@16
          |                     |
        e4(?)                 e3(78%)
          v                     v
        TRD@32  --e2(41%)-->  B2@32          （箭头方向 = "上游打赢下游"的那一比）

环一闭合，**第四条边就被前三条完全决定了**（Bradley-Terry：logit 胜率 = 潜在质量差，可加）：

    d4 = q(T32) - q(T16) = [q(T32)-q(B32)] + [q(B32)-q(B16)] + [q(B16)-q(T16)] = d2 + d3 - d1

这件事的分量：一年来 32px 的全部杠杆都按 **(H_TRD)「TRD 随画布变差」** 设计、**全灭**。
若 d4 > 0，则我们的产物随画布**也在变好**，只是**没有 B2 变得快**——那 (H_TRD) 的原始形式就是假的。

================================================================ 三条限制，全部写在前面
(L1) **可加性/传递性**：判官在不同画布尺寸之间的比较必须落在同一把潜在尺子上。
     这台仪器**从没验过**。环的第四条边正是它的检验——直接量 e4，看落不落在预测区间里。
     → 这就是 `eval/trd_canvas.sh` 的理由。**在它回来之前，下面的 d4 只是预测，不是结果。**

(L2) **弃样口径会自己造出一个 d3**：四条边的判出率差得很远（e1 73%、e2 74%、**e3 只有 51%**）。
     判官若有"分不出就弃"的不敏感带，则**丢掉平局会把 logit 往外推**——判出率越低推得越狠。
     e3 恰恰是判出率最低、logit 最大的那条边，**这正是该假象的方向**。
     所以本文件**两把尺子都报**：
       (S1) 只算判出（与已发表三档同口径）；
       (S2) 把 inconsistent 记 0.5 分、分母固定 272（平局不丢）。
     两把尺子给出**两个不同的点预测**，e4 回来时对两个都检验。只在一把尺子上闭合 = 结论不稳。

(L3) **e4 不是"TRD 的画布增益"，它捆着检查点**：`eval/final_test.sh:18,22` 写着
     TRD16c 出自 **v8**、TRD32 出自 **v10**（`--ret_nname` 都是 100）。所以 d4 = 画布 + 代次。
     → 可以说的是"**我们交付的产物**从 16px 档到 32px 档变好了多少"（缺口问的正是交付物）；
       **不可以**说成"TRD 从画布拿到多少增益"。(L1) 的可加性检验不受这个混淆影响
       ——它检验的是判官对这四个具体产物是否传递，与产物怎么来的无关。

    python analysis/arch/tier_cycle.py
"""
import json
import os
import sys
from math import erfc, exp, log, sqrt

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from exact import binom_test, jeffreys        # noqa: E402

EXP = os.path.join(ROOT, "experiments")

EDGES = {
    "d1": ("judge_full_TRD16c_rr4_vs_B2_16.json", "TRD@16 vs B2@16   （已发表：16px 这档我们胜）"),
    "d2": ("judge_full_TRD32_rr4_vs_B2_32.json",  "TRD@32 vs B2@32   （已发表：32px 这档我们输）"),
    "d3": ("judge_full_B2_vs_B2up16_32.json",     "B2@32  vs B2@16   （`557a50e` 新量：画布增益）"),
}

# `eval/trd_canvas.sh`（预注册 `2f64dd1`）的产物；跑完 scp 进 experiments/ 后本脚本自动接上。
E4_FILE = "judge_full_TRD32_rr4_vs_TRD16cup_32.json"


def logit(p):
    return log(p / (1.0 - p))


def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))


def read_edge(fname, fail_as=None):
    """从逐对记录重算 (胜, 判出, 弃, 总)。不信任 JSON 的汇总字段——复核脚本的老规矩。

    `fail_as`：只有在 api_fail>0 时才允许非 None。把 API 失败的那几对**显式**当成
    'A' / 'B' / 'inconsistent' 算进去，用来做**最坏情况夹逼**——判决必须在三种赋值下
    全都一样，否则该臂作废。这不是放宽操作检验 (8)：(8) 照旧记为"未通过"，
    只是用一个不依赖缺失值的界把它的影响量死。总对数恒为 len(records)，(S2) 分母不变。
    """
    d = json.loads(open(os.path.join(EXP, fname), encoding="utf-8").read())
    r = d["records"]
    w = sum(x["verdict"] == "A" for x in r)
    n = sum(x["verdict"] in ("A", "B") for x in r)
    inc = sum(x["verdict"] == "inconsistent" for x in r)
    nf = sum(x["verdict"] is None for x in r)
    assert (d["a_wins"], d["decided"], d["inconsistent"], d["api_fail"]) == (w, n, inc, nf), \
        f"{fname}: 汇总与逐对不自洽"
    if nf:
        assert fail_as in ("A", "B", "inconsistent"), f"{fname}: api_fail={nf}，必须显式说明怎么算"
        if fail_as == "A":
            w, n = w + nf, n + nf
        elif fail_as == "B":
            n = n + nf
        else:
            inc = inc + nf
    return w, n, inc, len(r)


def scale_decided(w, n, inc, tot):
    """(S1) 只算判出：m = w/n，二项方差。"""
    m = w / n
    return m, 1.0 / (n * m * (1.0 - m))


def scale_tiehalf(w, n, inc, tot):
    """(S2) 平局记 0.5 分、分母固定为总对数。逐对得分 ∈ {0, 0.5, 1} 的均值。"""
    m = (w + 0.5 * inc) / tot
    e2 = (w * 1.0 + inc * 0.25) / tot                 # E[s^2]
    var_mean = (e2 - m * m) / tot
    return m, var_mean / (m * (1.0 - m)) ** 2         # delta 法传到 logit


SCALES = [("S1 只算判出（已发表三档同口径）", scale_decided),
          ("S2 平局记 0.5、分母 272（平局不丢）", scale_tiehalf)]


def norm_p2(z):
    """双侧正态 p 值（纯标准库；服务器的 jzs_train 没有 scipy）。"""
    return erfc(abs(z) / sqrt(2.0))


def assumed_var(fn, r, p, tot=272):
    """造一份"判出率 r、判出里胜率 p"的假想 e4 计数，算它在某把尺子上的 logit 方差。

    只用于 e4 回来**之前**估这个检验的分辨率。e4 一落地就改用它自己的真实计数。
    """
    n = r * tot
    return fn(p * n, n, tot - n, tot)[1]


def main():
    raw = {}
    print("四个格子的三条已知边（w 由逐对记录重算）：\n")
    for k, (fname, desc) in EDGES.items():
        w, n, inc, tot = read_edge(fname)
        raw[k] = (w, n, inc, tot)
        lo, hi = jeffreys(w, n)
        print(f"  {k}  {desc}")
        print(f"      判出里 {w}/{n} = {w / n:.1%}  [{lo:.0%},{hi:.0%}]  p={binom_test(w, n):.3g}"
              f"   |   判出率 {n}/{tot} = {n / tot:.0%}（弃 {inc}）")

    preds = {}
    for label, fn in SCALES:
        est, var = {}, {}
        for k in EDGES:
            m, v = fn(*raw[k])
            est[k], var[k] = logit(m), v
        d4 = est["d2"] + est["d3"] - est["d1"]
        se = sqrt(var["d2"] + var["d3"] + var["d1"])
        lo, hi = d4 - 1.96 * se, d4 + 1.96 * se
        swing = est["d1"] - est["d2"]
        preds[label] = (d4, lo, hi, se, swing, fn)
        print("\n" + "=" * 78)
        print(f"【{label}】  单位 logit，正数 = 画布变大时变好\n")
        print(f"  B2  的画布增益        d3 = {est['d3']:+.3f} ± {sqrt(var['d3']):.3f}   （实测）")
        print(f"  我们交付物的档间增益  d4 = {d4:+.3f} ± {se:.3f}   "
              f"（**环推出来的，尚未实测**；按 (L3) 捆着 v8→v10）")
        print(f"  三档的落差       d1 - d2 = {swing:+.3f} ± {sqrt(var['d1'] + var['d2']):.3f}"
              f"   （恒等于 d3 - d4）")
        print(f"\n  -> `eval/trd_canvas.sh` 的点预测：胜率 {sigmoid(d4):.1%}"
              f"，95% 预测区间 [{sigmoid(lo):.1%}, {sigmoid(hi):.1%}]"
              f"；区间不含 50%？{'是' if lo > 0 else '否'}")

    print("\n" + "=" * 78)
    print("两把尺子的点预测放在一起（这就是 (L2) 说的「结论稳不稳」）：\n")
    for label, (d4, lo, hi, se, swing, fn) in preds.items():
        print(f"  {label[:2]}  {sigmoid(d4):5.1%}   [{sigmoid(lo):.1%}, {sigmoid(hi):.1%}]")
    print("\n  两个预测差得越远，说明「三档」这个读数越依赖弃样口径本身。"
          "\n  e4 回来后对**两把尺子各检验一次**：都判「未被证伪」才算环闭合。")

    print("\n" + "=" * 78)
    if os.path.exists(os.path.join(EXP, E4_FILE)):
        closure(preds)
    else:
        resolution(preds)
    return 0


def resolution(preds):
    """e4 未落地时：把闭合判据和它的分辨率先说死。"""
    print("【闭合判据（e4 未见时定死）】\n")
    print("  δ = logit(e4 实测) − d4(环推出)；SE(δ) = sqrt( var(d4 预测) + var(e4 自己) )。")
    print("  |z| = |δ| / SE(δ) < 1.96 → **环未被证伪**；≥ 1.96 → **环不闭合**。两把尺子各判一次。\n")
    print("  ⚠ 上面印的 [lo,hi] 是 **d4 这个参数的 CI，不含 e4 自己的抽样噪声**。")
    print("     拿实测点直接跟它比会系统性偏向判「不闭合」。**闭合一律用上面的 z 判，不用那个区间。**\n")
    print("  分辨率：80% 功效下能查出的最小不可加量 = 2.80 × SE(δ)。")
    print("  e4 的判出率未知，用已观测到的两个 32px 判出率（e2 74%、e3 51%）把它夹住：\n")
    print("    尺子  假设判出率   SE(δ)    最小可查出的不可加  （折成胜率偏离预测）    三档落差本身")
    for label, (d4, lo, hi, se, swing, fn) in preds.items():
        for r in (0.74, 0.51):
            vm = assumed_var(fn, r, sigmoid(d4))
            set_ = sqrt(se * se + vm)
            mde = 2.80 * set_
            print(f"    {label[:2]}      {r:.0%}      {set_:.3f}        {mde:+.3f} logit"
                  f"        ±{abs(sigmoid(d4 + mde) - sigmoid(d4)):.0%}pp"
                  f"            {swing:+.3f}")
    print("\n  ⚠⚠ **最小可查出量和三档落差本身同量级**（S1：约 ±0.9 vs 0.65）。")
    print("     所以 e4 落在区间内**只能说「没被证伪」，不能说「可加性成立」**——")
    print("     这个检验分辨不了「完全可加」与「不可加得和整个缺口一样大」。**结论必须这么写。**")
    print("\n  （另：e4 与三条已知边是各自独立的提问，但四条边共用同一批材质，")
    print("   材质级的相关会让真实 SE 略大于上表 → 上表偏乐观，方向对我们不利那边。）")


def closure(preds):
    """e4 已落地：按上面定死的判据判，两把尺子各一次；API 失败的对做最坏情况夹逼。"""
    d = json.loads(open(os.path.join(EXP, E4_FILE), encoding="utf-8").read())
    nf = d["api_fail"]
    print(f"【e4 实测】{E4_FILE}")
    if nf:
        bad = [x["material"] for x in d["records"] if x["verdict"] is None]
        print(f"  ⛔ 操作检验 (8) **未通过**：api_fail = {nf}（{', '.join(bad)}），预注册要求 0。")
        print(f"     不放宽判据——改为把这 {nf} 对分别当成 A/B/平局各算一遍，"
              f"判决在三种赋值下全一致才作数。\n")
    assigns = ["A", "inconsistent", "B"] if nf else [None]
    name = {"A": "全算 A 胜（对 A 最有利）", "inconsistent": "全算平局", "B": "全算 A 负（最不利）",
            None: "无缺失"}
    closed = []          # 每种赋值下：两把尺子是否都未被证伪
    for fa in assigns:
        w, n, inc, tot = read_edge(E4_FILE, fail_as=fa)
        lo, hi = jeffreys(w, n)
        print(f"  ── {name[fa]} ──")
        print(f"     TRD@32 vs TRD16c↑@32：判出里 {w}/{n} = {w / n:.1%}  [{lo:.0%},{hi:.0%}]"
              f"  p={binom_test(w, n):.3g}   |   判出率 {n}/{tot} = {n / tot:.0%}（弃 {inc}）")
        oks = []
        for label, (d4, _lo, _hi, se, swing, fn) in preds.items():
            m, vm = fn(w, n, inc, tot)
            delta = logit(m) - d4
            set_ = sqrt(se * se + vm)
            z = delta / set_
            ok = abs(z) < 1.96
            oks.append(ok)
            print(f"     {label[:2]}  环推 {d4:+.3f}±{se:.3f}  实测 {logit(m):+.3f}±{sqrt(vm):.3f}"
                  f"  δ={delta:+.3f}±{set_:.3f}  z={z:+.2f}  p={norm_p2(z):.3g}"
                  f"  → {'未被证伪' if ok else '**不闭合**'}")
        closed.append(all(oks))
        print()
    if all(closed):
        v = "两把尺子在所有赋值下都未证伪 → **环闭合**（在本检验的分辨率内）"
    elif not any(closed):
        v = "所有赋值下都至少有一把尺子判不闭合 → **环不闭合**，三档不能并排读"
    else:
        v = "⛔ 判决随缺失对的赋值翻转 → **该臂作废**，必须补测那几对"
    print("  【判决】" + v)
    print("  ⚠ 「未被证伪」不等于「可加性成立」：本检验的最小可查出量与三档落差同量级。")


if __name__ == "__main__":
    sys.exit(main())
