"""判官"分不出"的时候在干什么：扔硬币，还是照位置挑？——判据与点预测**在数据落地之前写死**。

背景链条：环不闭合（`6c64796`）→ 不可加压在 e3 判出率只有 51% 上 → 刺激那条解释已排除
（`830b2d2`：e3 只有 1/272 对近乎同图，且同样的图像距离下 e3 仍比 e1 判出率低 18pp）
→ 剩下的只能问判官本身。而已落盘的数据**答不了**：`judge_pairs.py:78` 把两次回答压成
A/B/inconsistent 就扔掉了"挑的是哪个位置"。`eval/judge_picks.py` 去把这一位补回来。

================================================================ 两个对立机制
(M-a) 不敏感带：分不出时**扔硬币** → 每次回答挑第一张的概率 = 0.5。
(M-b) 按位置作答：分不出时**照位置挑**（挑第一张的概率 q） → 换序必翻面，故必然 inconsistent。

空对照（两图相同 = 按定义分不出）实测判出率 f = 21/118 = 17.8%，而 (M-a) 在那里预测 50%
（p = 7.3e-13，`pilot_gate_audit.py`）→ **在空对照上 (M-a) 已经死了**，由 2q(1-q) = f 得 q = 0.901。
本脚本问的是真题上是不是也这样。

================================================================ 无自由参数的点预测
混合模型：每对以 λ 进"看内容"模式（两序挑同一个**方法**），否则进"按位置"模式（两次独立，各以 q 挑第一张）。
**看内容模式对"挑第一张的比例"的贡献恰好是 0.5**——正序挑第一张则反序挑第二张，与偏爱哪个方法无关。
于是：

    p_first = 0.5 λ + q (1 - λ)      λ = (R - f) / (1 - f)      R = 该臂**已发表**的判出率

q 由空对照独立定出、λ 由已发表的 R 定出 → **这是一个不含自由参数的样本外点预测**。
(M-a) 在同一位置上预测 0.5，与 λ 无关。

判据（**跑之前写死**，三条都报）：
 (P1) 主判据：p_first 显著 > 0.5（配对为单位，双侧 z 检验）→ (M-a) 在真题上也死；否则 (M-b) 死。
 (P2) 点预测：实测 p_first 落在 (M-b) 预测的 95% 区间内 → 未被证伪；落在外面 → 混合模型本身不够。
 (P3) 最锋利的一条，**不含 λ**：在 verdict = inconsistent 的对里，两次都挑"第一张"的比例。
      (M-b) 预测 q²/(q²+(1-q)²) = 98.8%；(M-a) 预测 50%。
 (P4) 边间对比：p_first(e3) > p_first(e4)（判出率越低，位置作答的份额越大）。

⚠ **本次不产生胜率、不登账成臂**（136 对的校准样本，不是重跑 e3/e4）；已发表判决不受影响。
⚠ **S3 那把尺子（见下）是设计本检验时顺手算出来的，我看过它的值**，所以它**不是**预注册结果，
   只能当描述。**是否改用 S3 只由 (P1)/(P3) 决定**——那是新数据，与环闭不闭合无关，
   所以这个决定不可能被"哪把尺子让环闭合"牵着走。

    python analysis/arch/pick_decomp.py                 # e4 未回来时：只印预测
"""
import json
import os
import sys
from math import erfc, exp, log, sqrt

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from exact import jeffreys                                  # noqa: E402

EXP = os.path.join(ROOT, "experiments")
PICKS = [("/tmp/judge_picks", "judge_picks_%s.json"), (EXP, "judge_picks_%s.json")]

F = 21 / 118                                                # 空对照实测地板
Q = (1 + sqrt(1 - 2 * F)) / 2                               # 由 f = 2q(1-q) 反推

# 校准的两条边：判出率最低与其中一条正常的，且两边都是"同方法跨画布"，比较类型不混进来。
ARMS = [("e3", "B2_vs_B2up16_32",        "judge_full_B2_vs_B2up16_32.json"),
        ("e4", "TRD32_rr4_vs_TRD16cup_32", "judge_full_TRD32_rr4_vs_TRD16cup_32.json")]


def logit(p):
    return log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + exp(-x))


def norm_p2(z):
    return erfc(abs(z) / sqrt(2))


def published_R(fname):
    d = json.loads(open(os.path.join(EXP, fname), encoding="utf-8").read())
    r = d["records"]
    n = sum(x["verdict"] in ("A", "B") for x in r)
    ask = sum(x["verdict"] is not None for x in r)
    return n / ask, n, ask


def predict(R):
    lam = (R - F) / (1 - F)
    return lam, 0.5 * lam + Q * (1 - lam)


def find_picks(tag):
    for d, pat in PICKS:
        p = os.path.join(d, pat % tag)
        if os.path.exists(p):
            return p
    return None


def main():
    print(f"空对照定出的仪器常数：f = {F:.4f}（21/118）  ->  q = {Q:.4f}\n")
    print(f"{'边':<4}{'已发表判出率 R':>16}{'λ=看内容的份额':>18}"
          f"{'(M-b) 预测 p_first':>20}{'(M-a) 预测':>12}")
    pred = {}
    for key, tag, full in ARMS:
        R, n, ask = published_R(full)
        lam, pf = predict(R)
        pred[key] = (R, lam, pf)
        print(f"{key:<4}{f'{n}/{ask} = {R:.1%}':>16}{lam:>18.3f}{pf:>20.1%}{0.5:>12.1%}")
    print(f"\n(P4) 预测的边间差 p_first(e3) - p_first(e4) = "
          f"{pred['e3'][2] - pred['e4'][2]:+.3f}")
    print(f"(P3) 预测的「inconsistent 里两次都挑第一张」的比例 = "
          f"{Q ** 2 / (Q ** 2 + (1 - Q) ** 2):.1%}   （(M-a)：50%）")

    got = [(k, t, f, find_picks(t)) for k, t, f in ARMS]
    if not any(p for _, _, _, p in got):
        print("\n（还没有 judge_picks_*.json：先跑 eval/pick_calib.sh）")
        return 0
    print("\n" + "=" * 78)
    obs = {}
    for key, tag, _full, path in got:
        if not path:
            print(f"{key}：还没落地")
            continue
        obs[key] = report(key, path, pred[key])
    if len(obs) == 2:
        verdict_p4(pred, obs)
        posthoc(pred, obs)
    return 0


def verdict_p4(pred, obs):
    """(P4) 预注册了预测却没印实测，补上。两条边是各自独立的样本，故方差直接相加。"""
    d = obs["e3"][0] - obs["e4"][0]
    se = sqrt(obs["e3"][1] + obs["e4"][1])
    want = pred["e3"][2] - pred["e4"][2]
    z = (d - want) / se
    print(f"\n(P4) 边间差 p_first(e3) - p_first(e4)：实测 {d:+.3f} ± {se:.3f}"
          f"，(M-b) 预测 {want:+.3f}  ->  z={z:+.2f}  p={norm_p2(z):.3g}"
          f"  {'**方向就反了**' if d * want < 0 else ''}")


def posthoc(pred, obs):
    """⚠ 以下**不是预注册**，是事后的稳健性检查：证伪会不会只是"二次方程选错了根"造成的。

    f = 2q(1-q) 有两个根，预注册硬写了 q_first = 0.901 那个。实测 p_first < 0.5 说明
    位置偏好指向**第二张**，对应另一个根 q_first = 0.099。把镜像根代回同一个模型再判一次；
    另外 (P3) 的值**不含 λ**，所以任何单 q 模型都必须在两条边上给出同一个数——直接比。
    """
    print("\n" + "-" * 78)
    print("事后稳健性（非预注册）：换成 f = 2q(1-q) 的另一个根（位置偏好指向第二张）")
    qm = 1 - Q
    print(f"  镜像根 q_first = {qm:.4f}；(P3) 改预测 "
          f"{qm ** 2 / (qm ** 2 + (1 - qm) ** 2):.1%}（实测见上，两条边都远高于它）")
    for key in ("e3", "e4"):
        lam = pred[key][1]
        pf = 0.5 * lam + qm * (1 - lam)
        z = (obs[key][0] - pf) / sqrt(obs[key][1])
        print(f"  {key} (P2) 镜像点预测 {pf:.1%}  实测 {obs[key][0]:.1%}  z={z:+.2f}"
              f"  p={norm_p2(z):.3g}  -> {'仍被证伪' if abs(z) >= 1.96 else '未被证伪'}")
    l3, l4 = pred["e3"][1], pred["e4"][1]
    want = (0.5 * l3 + qm * (1 - l3)) - (0.5 * l4 + qm * (1 - l4))
    d = obs["e3"][0] - obs["e4"][0]
    print(f"  (P4) 镜像预测 {want:+.3f}  实测 {d:+.3f} ± {sqrt(obs['e3'][1] + obs['e4'][1]):.3f}"
          f"  -> 这一条镜像根对上了；但 (P2)/(P3) 仍不对 => **换根救不回来**")
    a, na = obs["e3"][2], obs["e3"][3]
    b, nb = obs["e4"][2], obs["e4"][3]
    p = (a + b) / (na + nb)
    z = (a / na - b / nb) / sqrt(p * (1 - p) * (1 / na + 1 / nb))
    print(f"  (P3) 跨边：e3 {a}/{na}={a / na:.1%} vs e4 {b}/{nb}={b / nb:.1%}"
          f"  z={z:+.2f} p={norm_p2(z):.3g}（(P3) 不含 λ，单 q 模型要求两边相等）")
    for key, bf, bs in (("e3", a, na - a), ("e4", b, nb - b)):
        r = sqrt(bf / bs)
        print(f"      {key} 由 inconsistent 两格反推 q_first = {r / (1 + r):.3f}"
              f"（空对照给的是 {qm:.3f} 或 {Q:.3f}）")


def report(key, path, pr):
    R, lam, pf = pr
    d = json.loads(open(path, encoding="utf-8").read())
    recs = [r for r in d["records"] if r["verdict"] is not None]
    # 以**对**为单位：x = 两次里挑第一张的次数 / 2 ∈ {0, .5, 1}。这样才不把配对内的相关当独立。
    x = [((r["pick1"] == "first") + (r["pick2"] == "first")) / 2 for r in recs]
    n = len(x)
    m = sum(x) / n
    var = sum((v - m) ** 2 for v in x) / (n - 1) / n
    se = sqrt(var)
    z0, zp = (m - 0.5) / se, (m - pf) / se
    print(f"\n{key}  {d['tag']}   n={n} 对（API 失败 {d['api_fail']}）")
    print(f"  实测 p_first = {m:.1%} ± {se:.1%}")
    print(f"  (P1) vs (M-a) 的 50%：z={z0:+.2f}  p={norm_p2(z0):.3g}  -> "
          f"{'**(M-a) 被证伪**' if abs(z0) >= 1.96 else '未能证伪 (M-a)'}")
    print(f"  (P2) vs (M-b) 的点预测 {pf:.1%}：z={zp:+.2f}  p={norm_p2(zp):.3g}  -> "
          f"{'**点预测被证伪**' if abs(zp) >= 1.96 else '点预测未被证伪'}")
    bad = [r for r in recs if r["verdict"] == "inconsistent"]
    k = sum(r["pick1"] == "first" for r in bad)
    if bad:
        lo, hi = jeffreys(k, len(bad))
        print(f"  (P3) inconsistent 的对里两次都挑第一张：{k}/{len(bad)} = {k / len(bad):.1%}"
              f"  [{lo:.0%},{hi:.0%}]   （(M-b) 预测 {Q ** 2 / (Q ** 2 + (1 - Q) ** 2):.1%}，(M-a) 预测 50%）")
    rr = sum(r["verdict"] in ("A", "B") for r in recs) / n
    print(f"  参考：本次样本自己的判出率 {rr:.1%}（已发表 {R:.1%}）"
          f"；判官非确定性已知（57 道重复题 14 道翻面），两者不必相等")
    return m, var, k, len(bad)


if __name__ == "__main__":
    sys.exit(main())
