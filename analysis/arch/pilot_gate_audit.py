"""零 API、零 GPU：把 20 条判官试点的**空对照**汇总，量出这台仪器真正的地板。

为什么写：每条臂开跑前的试点都附 5 对"两边同一张图"的空对照，用来给可解率定地板。
20 条臂跑下来，这 100 对空对照**从没被合并看过一次**——单条臂只有 5 对，看不出名堂。
合起来能回答两件之前只能靠推理的事：

1. **地板不是 50%。** 论文期 `crop_scale_study.py` 的推导是：判官若不看内容、以概率 q 选第一张，
   跨序一致率 = 2q(1-q) <= 0.5，q=0.5 时取到 0.5。**那是上界，不是这台仪器的实测值。**
   去重后 90 题实测 **15.6%** → 反推 q ~ 0.91：`judge_pairs.py` 的判官在两图相同时
   **几乎总是挑同一个位置**，换序就翻面，于是判成 inconsistent。
2. 于是"可解率低"要分两种读法：低到 16% 附近 = 判官只在按位置作答（真的没信息）；
   60% 出头 = **远高于地板**，只是没到我们为了保功效而画的 65% 那条线。
   **把 62% 写成"判官没分辨力"是错的**，脚本里那句提示词要按这个改。

⚠ **必须去重（2026-09-14 发现，此前的 "100 对 / 12%" 是把重复当独立了）**：
`judge_pairs.py:122-124` 的空对照用的是 **A 臂自己的瓦片**（`ta,ta`），选题靠 `default_rng(0)`
的固定排列。于是**同一张参照臂 + 同样的可比对张数**下，不同的 B 臂得到的是**同一批 5 道题**。
16px 消融表五条臂共 25 条空对照记录，其实只有 **5 道不同的题**（对号 70/258/202/5/264）。
本脚本按 (A 臂, 对号) 去重，重复的只取第一次。副产物：15 道被重复问过的题里 **3 道前后答案不一致**
→ **这台仪器不是确定性的**（论文期那句"温度 0 下判官确定"是另一台仪器上量的，不能搬过来）。

⚠ 本脚本一个胜负字段都不读（只读 kind/answered/resolved，full 那边只读 verdict 是否 in A/B
   ——即"判没判出"，不读判给了谁），不会把试点变成偷看结果。

---------------------------------------------------------------------------
**第二节（2026-09-14 追加）：门的假阴性率。**

`b2_canvas` 是**连续第三条**死在这道门上的臂（前两条 `nod32`、`tier_anchor` 的 B1@24）。
上面第一节已经说清"不过门 != 判官没分辨力"，但没回答**这道门到底误杀多少**。读代码先拿到两个
与数据无关的事实：

  * 试点对是 `default_rng(0).permutation(len(pairs))[:15]`（`judge_pairs.py:118-119`），种子写死
    → **池子大小相同的臂抽到的是同一批 15 个材质**（E_mat 272 对全是 5,31,70,84,...）。
    各臂试点**不是独立抽样**；这 15 个若偏难，每条臂被同样地罚一次。
  * full 对每一对**重新问一遍**（`:159-161` 又调了一次 `ask_pair`，同样的左右顺序），
    而试点的 `resolved` 与 full 的"判出"是同一个定义（两序一致）
    → 那 15 对在 full 里被**重测**了，可以直接量判官自身的抖动。

跑之前写死四个读法：

 (M1) 试点样本偏倚：在同时有试点与 full 的臂上，把 full 的判出率拆成「试点那 15 对」vs「其余对」。
      若那批**不更难**（区间重叠），"门被一批倒霉材质带偏"死掉，剩下的只能是功效问题。
 (M2) 重测一致性：同一对、同一顺序问两遍，判出与否是否一致；合并做精确 McNemar。
      这一项量的是**与 n 无关**的那部分抖动——它若很大，把 n 加大也救不回全部。
 (M3) 功效曲线：真值 p 时 P(15 抽中 ≥10)（10/15=66.7% 才够 ≥65%），在 full 实测的判出率上读假阴性。
 (M4) 要把假阴性压到 ≤10%，n 得多大。

**无论结果如何都成立的纪律**（先写在这里，免得被结果牵着走）：
  * **不许拿本节的结论给已经跑过的臂放宽门槛重跑**——那是在门上做 p-hacking。
    `nod32` / `B1@24` / `b2_canvas` 三条的"无判决"是终局；要重开必须另行预注册**新的**门。
  * 本节只用来设计**今后**的预注册（`d0345de` 已记：门要加在「对」上，或把 n 提到 30+）。

    python analysis/arch/pilot_gate_audit.py
"""
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from exact import binom_test, jeffreys     # noqa: E402

FIELDS = ("kind", "answered", "resolved")   # 白名单：verdict 之类一律不碰


def main():
    files = sorted(glob.glob(str(ROOT / "experiments" / "judge_pilot_*.json")))
    if not files:
        print("没有试点 JSON（净克隆里应当有，见 .gitignore 的豁免）")
        return 1
    R = RA = 0
    rows = []
    seen = {}          # (A 臂, 对号) -> 第一次的 resolved；空对照跨臂重复，必须去重
    repeat = []        # 同一道题被重复问到的答案，用来看这台仪器确不确定
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        arm_a = d["tag"].split("_vs_")[0]
        recs = [{k: x[k] for k in FIELDS} | {"pair": x["pair"]} for x in d["records"]]
        r = [x for x in recs if x["kind"] == "real" and x["answered"]]
        n = [x for x in recs if x["kind"] == "null" and x["answered"]]
        rr, nn = sum(x["resolved"] for x in r), sum(x["resolved"] for x in n)
        R += rr; RA += len(r)
        for x in n:
            key = (arm_a, x["pair"])
            if key in seen:
                repeat.append((seen[key], x["resolved"]))
            else:
                seen[key] = x["resolved"]
        rows.append((d["tag"], rr, len(r), nn, len(n), d["min_rate"], d["pass"]))
    N, NA = sum(seen.values()), len(seen)

    print(f"{'臂':52s} {'真题':>9s} {'空对照':>8s}  门槛  过?")
    for tag, rr, rn, nn, nnn, mr, ok in rows:
        print(f"{tag:52s} {rr:3d}/{rn:<3d}{rr / rn:5.0%} {nn:2d}/{nnn:<2d}{nn / nnn:4.0%}  {mr:.2f}  "
              f"{'OK' if ok else 'X'}")

    print(f"\n合并真题   {R}/{RA} = {R / RA:.1%}  Jeffreys [{jeffreys(R, RA)[0]:.1%},{jeffreys(R, RA)[1]:.1%}]")
    lo, hi = jeffreys(N, NA)
    print(f"合并空对照 {N}/{NA} = {N / NA:.1%}  Jeffreys [{lo:.1%},{hi:.1%}]  "
          f"vs 50% 的二项 p = {binom_test(N, NA):.3g}   "
          f"（已按 (A 臂, 对号) 去重，丢掉 {len(repeat)} 条重复记录）")
    flip = sum(x != y for x, y in repeat)
    print(f"重复问过的 {len(repeat)} 次里，与第一次答案不一致的有 {flip} 次 "
          f"-> 这台仪器**不是确定性的**，别把论文期那句'温度 0 下判官确定'搬过来")
    print("\n读法：空对照 = 两边同一张图，判官若按内容作答无从判起。实测远低于 2q(1-q) 的上界 0.5，")
    print(f"     说明它在平局上几乎总按位置作答（q ~ {(1 + (1 - 2 * N / NA) ** 0.5) / 2:.2f}）"
          f"-> **这台仪器的可解率地板约 {N / NA:.0%}，不是 50%**。")

    fails = [x for x in rows if not x[6]]
    if fails:
        print(f"\n未过门槛的 {len(fails)} 条（都**没有** full，结论只能是'没到我们要的功效'，不是'判官没分辨力'）：")
        for tag, rr, rn, nn, nnn, mr, _ in fails:
            print(f" - {tag}: 真题 {rr}/{rn} = {rr / rn:.0%}，空对照 {nn}/{nnn}；"
                  f"差 {int(mr * rn - rr) + (1 if mr * rn > rr else 0)} 条就过线")
    print("\n注意：n=15 时门槛的分辨力只有 6.7pp——pb05 以 10/15=67% 过线、pb10 以 8/13=62% 落线，")
    print("  两者相差不到一条记录。**别把'过/不过'读成两种性质不同的结果。**")
    gate_power(files, N / NA)
    return 0


def binom_pmf_ge(p, n, thr):
    """真值 p 时，n 抽里可解数达到 ceil(thr*n) 的概率（精确二项，无 scipy）。"""
    import math
    need = math.ceil(thr * n - 1e-9)
    return sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(need, n + 1))


def gate_power(pilot_files, N_FLOOR):
    """第二节：这道门误杀多少。只读'判没判出'，不读判给了谁。"""
    pilots = {}
    for f in pilot_files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        pilots[d["tag"]] = d
    fulls = {}
    for f in sorted(glob.glob(str(ROOT / "experiments" / "judge_full_*.json"))):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        fulls[d["tag"]] = {r["pair"]: r["verdict"] in ("A", "B") for r in d["records"]}
    both = sorted(set(pilots) & set(fulls))
    print(f"\n{'=' * 78}\n第二节：门的假阴性率（试点 {len(pilots)} 条，full {len(fulls)} 条，两者都有 {len(both)} 条）")

    print("\n== (M1) 试点固定抽到的那批对，在 full 里更难判吗 ==")
    ik = inn = ok_ = onn = narm = 0
    for tag in both:
        idx = {r["pair"] for r in pilots[tag]["records"] if r["kind"] == "real"}
        fv = fulls[tag]
        a = sum(v for k, v in fv.items() if k in idx); an = sum(1 for k in fv if k in idx)
        b = sum(v for k, v in fv.items() if k not in idx); bn = sum(1 for k in fv if k not in idx)
        if not an or not bn:
            continue
        ik += a; inn += an; ok_ += b; onn += bn; narm += 1
        print(f"  {tag:50s} 试点那批 {a:2d}/{an:<3d}={a / an:4.0%}   其余 {b:3d}/{bn:<3d}={b / bn:4.0%}")
    r1, r2 = ik / inn, ok_ / onn
    lo1, hi1 = jeffreys(ik, inn)
    lo2, hi2 = jeffreys(ok_, onn)
    print(f"  合并：试点那批 {ik}/{inn} = {r1:.1%} [{lo1:.0%},{hi1:.0%}]   "
          f"其余 {ok_}/{onn} = {r2:.1%} [{lo2:.0%},{hi2:.0%}]   差 {(r1 - r2) * 100:+.1f}pp")
    print(f"  -> {'区间重叠，读作「不更难」：门的问题不在选题，在功效' if lo1 < hi2 and lo2 < hi1 else '区间不重叠：选题本身偏倚，需换种子'}")

    print("\n== (M2) 同一对同一顺序问两遍（试点一遍、full 一遍），判出与否一致吗 ==")
    yy = yn = ny = nn2 = 0
    for tag in both:
        fv = fulls[tag]
        for r in pilots[tag]["records"]:
            if r["kind"] != "real" or r["pair"] not in fv:
                continue
            a, b = bool(r["resolved"]), fv[r["pair"]]
            yy, yn, ny, nn2 = (yy + (a and b), yn + (a and not b), ny + (b and not a),
                               nn2 + (not a and not b))
    tot = yy + yn + ny + nn2
    disc = yn + ny
    print(f"  两遍都判出 {yy}   都判不出 {nn2}   只有试点判出 {yn}   只有 full 判出 {ny}")
    print(f"  一致率 {(yy + nn2) / tot:.1%}（n={tot}）；**重问一次就翻面 {disc}/{tot} = {disc / tot:.0%}**")
    if disc:
        pm = binom_test(yn, disc)
        print(f"  精确 McNemar（{yn}:{ny}）p={pm:.3g} -> "
              f"{'方向无偏，是纯抖动' if pm > 0.05 else '两遍系统性不同，需追因'}")
    print(f"  含义：这 {disc / tot:.0%} 与 n 无关——加大试点也消不掉，只能靠重复提问取多数")

    print("\n== (M3) 功效：真值判出率 p 时，n=15 的试点过 65% 门的概率 ==")
    print("  （只算 min_rate 一条；空对照 > 那一条是另加的，只会更难过）")
    for p in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
        q = binom_pmf_ge(p, 15, 0.65)
        print(f"   p={p:.0%} -> 过门 {q:5.0%}   假阴性 {1 - q:5.0%}")
    obs = (ik + ok_) / (inn + onn)
    print(f"\n  full 实测总体判出率 = {obs:.1%}（{ik + ok_}/{inn + onn}，{narm} 条**已经过了门**、且试点与 full 可拆分的臂，"
          f"故这是有偏上估）；在该真值上假阴性 ≈ {1 - binom_pmf_ge(obs, 15, 0.65):.0%}")

    print("\n== (M4) 同一个 65% 门槛，n 要多大才能把假阴性压到 ≤10% ==")
    for p in (0.70, 0.75, 0.80):
        n = next((n for n in range(15, 600) if 1 - binom_pmf_ge(p, n, 0.65) <= 0.10), None)
        print(f"   真值 {p:.0%}：n ≥ {n}" if n else f"   真值 {p:.0%}：600 以内做不到")
    print("   -> 门本来是为了在 full（272 对）之前省 API；若要它在 65% 上站得住就得问 120 对，"
          "\n      **那还不如直接跑 full**。65% 这条线是问不成的，不是 n 的问题。")

    print("\n== (M5) 同样 n=15 能撑起什么样的门：拿地板当零假设，而不是拿 65% ==")
    print(f"   零假设 = 第一节实测的地板 {N_FLOOR:.0%}（判官纯按位置作答）；备择 = 实测判出率 70%")
    print(f"   {'门槛 k/15':>10s} {'= 比率':>7s}   {'误放(p=地板)':>13s}   {'误杀(p=70%)':>12s}")
    best = None
    for k in range(4, 13):
        fp = binom_pmf_ge(N_FLOOR, 15, k / 15)
        fn = 1 - binom_pmf_ge(0.70, 15, k / 15)
        star = ""
        if fp <= 0.10 and fn <= 0.10 and best is None:
            best, star = k, "   <- 两种错都 ≤10%，最省的一条"
        print(f"   {'≥ ' + str(k) + '/15':>10s} {k / 15:6.0%}   {fp:12.1%}   {fn:11.0%}{star}")
    if best:
        print(f"   -> **n=15 能可靠回答的是「高于地板吗」（门槛 ≥{best}/15 = {best / 15:.0%}），"
              f"不是「高于 65% 吗」。**")
    print("   注意：这是给**今后**的预注册用的候选，本轮不改 judge_pairs.py 的默认值；")
    print("        换门槛必须另行预注册，且不许拿它去重跑已经被拦下的臂。")
    print("\n[!] 本节不改变任何已下的判决；已跑过的臂不许据此放宽门槛重跑。")


if __name__ == "__main__":
    raise SystemExit(main())
