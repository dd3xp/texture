#!/usr/bin/env python
"""(M57) 盲写判读器：模型在**同材质**上比它自己的 32px 训练料更靠"平铺两遍"那一侧吗？

零 GPU、零 API、零判官、零训练。判据全文在 docs/arch_progress.md 的 (M57) 预注册节，
本文件与那一节**同批提交、先于任何读数**。跑完判据一个字不许改。

================================ 仪器（一个字没改，(M22)/(M23) 那把） ================================
每张图与"**左上象限平铺两遍**"比：
  exact = 逐像素完全相同；mad = 每通道绝对差的均值（0-255）。
  【项目规矩】量图像差异只许用 mad，不许数"多少像素不相等"。
真人瓦片渲成像素的写法照抄 `train_pool_tiling.py`：`palette[idx]`。
mad 小 => 更靠"32px 就是 16px 平铺两遍"那一侧（纯平铺恰好 0，这是**精确零**，不必借任何门槛）。

================================ 配对（跑前定死） ================================
主键 = **材质名**。每个材质：
  MAD_gen(m)  = 该材质生成的 32px 图（n=2）的 mad 中位
  MAD_real(m) = 该材质在 **base 32px 训练池**（`extra=False`）里全部瓦片的 mad 中位
  d(m) = MAD_gen(m) - MAD_real(m)  <0 = 产物更靠平铺侧
⛔ 不含 `train_64to32.json`（64px 降采样，像素周期是预处理产物）与 `*_extra_packs_only`（只在远程）。

================================ 判据（主判决只有一条） ================================
M_real = 各材质 MAD_real 的中位；med_d = 各材质 d 的中位；frac_neg = d<0 的比例（d==0 不计入分母）。
  - **CANVAS_WASTED**：frac_neg >= 0.65  且  双侧精确符号检验 p < 0.01  且  med_d <= -0.5 * M_real
  - 否则 **CANVAS_SMALL**：方向或幅度任一不够 => 这条机制**量级上不足半个画布**，
    作为"要花一次重训的杠杆"就此关闭。
后缀：
  - `_FRAGILE`：按包 LOPO（只看材质数 >= 10 的包）任一去掉会翻转主判决 => 只能引方向。
  - `_CONFOUND`：标准化副读数 dn（mad / 该图自身像素标准差）的中位**符号与 med_d 相反**
    => 生成产物色彩起伏更小这个混杂没排除掉，只能引方向。

================================ 作废条件（按序先查，任一触发本轮无读数） ================================
 1. `VOID_NO_DATA`：配得上的材质 < 50，或 32px/16px 任一档产物张数为 0。
 2. `VOID_GEN_INCOMPLETE`：T32b 里存在**一个**材质没有 32px 产物。
 3. `VOID_RULER_TRIVIAL`：拿同一把尺子问我们自己的 **16px 产物**"是不是 8px 平铺两遍"，
    exact 率 >= 0.20 => 这把尺子在我们的产物上普遍触发，32px 读数不可解读。
    （门 0.20 照抄 (M23) 的 (OP1)：同一把尺子、同一个"非平凡"问题，⛔ 不是从别处借的门。）
 4. `VOID_COPY_LEAK`：32px 产物里**与同材质真人瓦片逐像素完全相同**的比例 >= 0.10
    => 检索/调色板记忆库把训练瓦片抄回来了，差值不能归给"模型用不用画布"。

================================ 读法（跑前写死，⛔ 越界的一条都不许） ================================
 1. ⛔ mad 不许换算成胜率／KID（(M51)：指标到判官的换算造不出来）、⛔ 不许当优化目标或挑配置的依据
    （项目禁令：结构门/各向异性永远不许当目标）。本轮只定位机制，不授权任何默认值改动。
 2. ⛔ 不许读成"32px 缺口找到了/解释了"；32px 准入条件（`6c64796`）一个字不松；⛔ 不开判官臂。
 3. ⚠ 已披露偏倚：T32b 是**训练材质**（模型见过那些瓦片），且生成用 `--pal_mode retrieve --xmodal`
    （调色板与结构范例都从记忆库检索）=> 偏向"产物更像真人瓦片"，即**对 CANVAS_WASTED 保守**。
    作废条件 4 只挡"逐像素抄回来"这一种极端。
 4. ⚠ `CANVAS_SMALL` 只许读成"这条机制在这把尺子上量级不足半个画布"，
    ⛔ 不许读成"模型会用画布"、⛔ 不许读成"32px 没问题"。

用法：
    python analysis/arch/m57_read_canvas.py --selftest
    python analysis/arch/m57_read_canvas.py --gen /tmp/m57_gen/m57t32 --out /tmp/m57_canvas.json
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

FRAC_NEG_GATE = 0.65
P_GATE = 0.01
MAG_GATE = 0.5          # med_d <= -MAG_GATE * M_real
MIN_PAIRED = 50
RULER_TRIVIAL_EXACT = 0.20
COPY_LEAK_GATE = 0.10
PACK_MIN_MATS = 10


# ------------------------------------------------------------------ 仪器
def readout(a):
    """a: [n,n,3] int16。返回 (exact, mad, mad_norm)。"""
    a = a.astype(np.int16)
    n = a.shape[0]
    if n % 2 or a.shape[1] != n:
        raise SystemExit(f"尺寸不是偶数方阵 {a.shape}")
    h = n // 2
    t = np.tile(a[:h, :h], (2, 2, 1))
    mad = float(np.abs(a - t).mean())
    sd = float(a.std())
    return bool(np.array_equal(a, t)), mad, mad / sd if sd > 1e-6 else 0.0


def binom_two_sided(k, n):
    """精确二项检验（p=0.5，双侧）。jzs_train 没有 scipy，用 math.comb。"""
    if n == 0:
        return 1.0
    tot = 2.0 ** n
    probs = [math.comb(n, i) / tot for i in range(n + 1)]
    obs = probs[k]
    return float(min(1.0, sum(p for p in probs if p <= obs * (1.0 + 1e-9))))


# ------------------------------------------------------------------ 判决（纯函数，供 selftest）
def decide(st):
    """st 里只放判据要的那几个数；返回 (verdict, why)。"""
    why = {}
    if st["n_paired"] < MIN_PAIRED or st["n_gen32"] == 0 or st["n_gen16"] == 0:
        return "VOID_NO_DATA", why
    if st["n_missing_gen"] > 0:
        return "VOID_GEN_INCOMPLETE", why
    if st["ruler16_exact_rate"] >= RULER_TRIVIAL_EXACT:
        return "VOID_RULER_TRIVIAL", why
    if st["copy_leak_rate"] >= COPY_LEAK_GATE:
        return "VOID_COPY_LEAK", why

    why["frac_neg_pass"] = st["frac_neg"] >= FRAC_NEG_GATE
    why["p_pass"] = st["p_sign"] < P_GATE
    why["mag_pass"] = st["med_d"] <= -MAG_GATE * st["M_real"]
    v = "CANVAS_WASTED" if all(why.values()) else "CANVAS_SMALL"
    if st.get("lopo_flip"):
        v += "_FRAGILE"
    if st["med_d"] != 0 and st["med_dn"] != 0 and \
            (st["med_d"] < 0) != (st["med_dn"] < 0):
        v += "_CONFOUND"
    return v, why


def pack_stats(d_by_mat, pack_of_mat):
    by = defaultdict(list)
    for m, d in d_by_mat.items():
        by[pack_of_mat[m]].append(d)
    rows = []
    for p, ds in sorted(by.items(), key=lambda kv: -len(kv[1])):
        nz = [x for x in ds if x != 0]
        rows.append({"pack": p, "n_mats": len(ds),
                     "med_d": float(np.median(ds)),
                     "frac_neg": (sum(1 for x in nz if x < 0) / len(nz)) if nz else None})
    return rows


# ------------------------------------------------------------------ 自检
def selftest():
    rng = np.random.default_rng(0)
    ok, bad = 0, []

    def chk(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            bad.append(name)

    # 仪器
    q = rng.integers(0, 256, (16, 16, 3)).astype(np.int16)
    pure = np.tile(q, (2, 2, 1))
    e, m, mn = readout(pure)
    chk("纯平铺 exact", e is True)
    chk("纯平铺 mad=0", m == 0.0)
    chk("纯平铺 madn=0", mn == 0.0)
    noise = rng.integers(0, 256, (32, 32, 3)).astype(np.int16)
    e2, m2, mn2 = readout(noise)
    chk("噪声非平铺", e2 is False and m2 > 30)
    chk("madn 有值", mn2 > 0)
    flat = np.zeros((32, 32, 3), np.int16)
    chk("常值图 madn=0 不炸", readout(flat)[2] == 0.0)
    half = np.concatenate([np.zeros((32, 16, 3), np.int16),
                           np.full((32, 16, 3), 200, np.int16)], axis=1)
    chk("半黑半白 mad>0", readout(half)[1] > 0)

    # 符号检验
    chk("p(50,100)=1", abs(binom_two_sided(50, 100) - 1.0) < 1e-9)
    chk("p(100,100) 小", binom_two_sided(100, 100) < 1e-29)
    chk("p(65,100)<0.01", binom_two_sided(65, 100) < 0.01)
    chk("p(60,100)>0.01", binom_two_sided(60, 100) > 0.01)
    chk("p(0,0)=1", binom_two_sided(0, 0) == 1.0)

    base = {"n_paired": 200, "n_gen32": 400, "n_gen16": 400, "n_missing_gen": 0,
            "ruler16_exact_rate": 0.01, "copy_leak_rate": 0.0,
            "frac_neg": 0.80, "p_sign": 1e-12, "med_d": -8.0, "M_real": 12.0,
            "med_dn": -0.3, "lopo_flip": []}
    chk("A 全过 => WASTED", decide(base)[0] == "CANVAS_WASTED")
    chk("B 幅度不够 => SMALL", decide({**base, "med_d": -3.0})[0] == "CANVAS_SMALL")
    chk("C 方向不够 => SMALL", decide({**base, "frac_neg": 0.55})[0] == "CANVAS_SMALL")
    chk("D p 不够 => SMALL", decide({**base, "p_sign": 0.2})[0] == "CANVAS_SMALL")
    chk("E 反向 => SMALL", decide({**base, "med_d": +8.0, "med_dn": +0.3})[0] == "CANVAS_SMALL")
    chk("F LOPO 翻 => _FRAGILE",
        decide({**base, "lopo_flip": ["p1"]})[0] == "CANVAS_WASTED_FRAGILE")
    chk("G 标准化反号 => _CONFOUND",
        decide({**base, "med_dn": +0.3})[0] == "CANVAS_WASTED_CONFOUND")
    chk("H 配对太少 => VOID_NO_DATA", decide({**base, "n_paired": 49})[0] == "VOID_NO_DATA")
    chk("I 缺图 => VOID_GEN_INCOMPLETE",
        decide({**base, "n_missing_gen": 1})[0] == "VOID_GEN_INCOMPLETE")
    chk("J 尺子平凡 => VOID_RULER_TRIVIAL",
        decide({**base, "ruler16_exact_rate": 0.20})[0] == "VOID_RULER_TRIVIAL")
    chk("K 抄回来 => VOID_COPY_LEAK",
        decide({**base, "copy_leak_rate": 0.10})[0] == "VOID_COPY_LEAK")
    chk("L 作废优先于主判据",
        decide({**base, "n_gen16": 0, "frac_neg": 0.99})[0] == "VOID_NO_DATA")

    rows = pack_stats({"a": -1.0, "b": -2.0, "c": +3.0}, {"a": "P", "b": "P", "c": "Q"})
    chk("逐包 2 行", len(rows) == 2 and rows[0]["n_mats"] == 2)
    chk("逐包 frac_neg", rows[0]["frac_neg"] == 1.0)

    print(f"selftest {ok}/{ok + len(bad)}" + (f"  失败：{bad}" if bad else "  全过"))
    return 1 if bad else 0


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gen", type=Path, default=Path("/tmp/m57_gen/m57t32"),
                    help="含 32/ 与 16/ 两个子目录")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    from PIL import Image
    sys.path.insert(0, str(ROOT / "model"))
    from tiles_data import load

    tset = json.loads((ROOT / "eval/prompt_sets_train.json").read_text(encoding="utf-8"))["T32b"]
    mats_set = [e["material"] for e in tset]

    # 真人侧（base 32px 训练池）
    real = load(32, "train", extra=False)
    real_by_mat = defaultdict(list)
    for s in real:
        real_by_mat[s["material"]].append(s)
    real_mad, real_madn, real_exact, pack_of_mat, real_px = {}, {}, {}, {}, {}
    for m, ss in real_by_mat.items():
        rs = [readout(s["palette"][s["idx"]]) for s in ss]
        real_mad[m] = float(np.median([r[1] for r in rs]))
        real_madn[m] = float(np.median([r[2] for r in rs]))
        real_exact[m] = [r[0] for r in rs]
        pk = defaultdict(int)
        for s in ss:
            pk[s["pack"]] += 1
        pack_of_mat[m] = max(sorted(pk), key=lambda p: pk[p])
        real_px[m] = [s["palette"][s["idx"]].astype(np.int16) for s in ss]

    # 生成侧
    def scan(d, expect_mats):
        out = defaultdict(list)
        files = sorted(Path(d).glob("*.png")) if Path(d).is_dir() else []
        for f in files:
            slug = f.stem.rsplit("_", 1)[0]
            out[slug + ".png"].append(np.asarray(Image.open(f).convert("RGB"), np.int16))
        return out, len(files)

    g32, n32 = scan(a.gen / "32", mats_set)
    g16, n16 = scan(a.gen / "16", mats_set)

    gen_mad, gen_madn, copy_hits, gen32_total = {}, {}, 0, 0
    min_mad_to_real = []
    for m in mats_set:
        if m not in g32:
            continue
        rs = [readout(x) for x in g32[m]]
        gen_mad[m] = float(np.median([r[1] for r in rs]))
        gen_madn[m] = float(np.median([r[2] for r in rs]))
        for x in g32[m]:
            gen32_total += 1
            rp = real_px.get(m, [])
            if rp:
                mm = min(float(np.abs(x - r).mean()) for r in rp)
                min_mad_to_real.append(mm)
                if any(np.array_equal(x, r) for r in rp):
                    copy_hits += 1

    # 16px 尺子非平凡（问"是不是 8px 平铺两遍"）
    r16 = [readout(x) for xs in g16.values() for x in xs]
    ruler16 = {"n": len(r16),
               "exact": int(sum(r[0] for r in r16)),
               "exact_rate": float(np.mean([r[0] for r in r16])) if r16 else 1.0,
               "mad_median": float(np.median([r[1] for r in r16])) if r16 else None}

    paired = [m for m in mats_set if m in gen_mad and m in real_mad]
    d_by_mat = {m: gen_mad[m] - real_mad[m] for m in paired}
    dn_by_mat = {m: gen_madn[m] - real_madn[m] for m in paired}
    ds = np.array([d_by_mat[m] for m in paired]) if paired else np.zeros(0)
    dns = np.array([dn_by_mat[m] for m in paired]) if paired else np.zeros(0)
    nz = ds[ds != 0]

    st = {"n_paired": len(paired), "n_gen32": n32, "n_gen16": n16,
          "n_missing_gen": sum(1 for m in mats_set if m not in gen_mad),
          "ruler16_exact_rate": ruler16["exact_rate"],
          "copy_leak_rate": (copy_hits / gen32_total) if gen32_total else 0.0,
          "frac_neg": float((nz < 0).mean()) if nz.size else 0.0,
          "p_sign": binom_two_sided(int((nz < 0).sum()), int(nz.size)),
          "med_d": float(np.median(ds)) if ds.size else 0.0,
          "med_dn": float(np.median(dns)) if dns.size else 0.0,
          "M_real": float(np.median([real_mad[m] for m in paired])) if paired else 0.0,
          "lopo_flip": []}

    # LOPO（只看材质数 >= PACK_MIN_MATS 的包）
    rows = pack_stats(d_by_mat, pack_of_mat)
    big = [r["pack"] for r in rows if r["n_mats"] >= PACK_MIN_MATS]
    v0, _ = decide({**st, "lopo_flip": []})
    for p in big:
        keep = [m for m in paired if pack_of_mat[m] != p]
        if len(keep) < MIN_PAIRED:
            continue
        sub = np.array([d_by_mat[m] for m in keep])
        subn = np.array([dn_by_mat[m] for m in keep])
        nz2 = sub[sub != 0]
        s2 = {**st, "n_paired": len(keep),
              "frac_neg": float((nz2 < 0).mean()) if nz2.size else 0.0,
              "p_sign": binom_two_sided(int((nz2 < 0).sum()), int(nz2.size)),
              "med_d": float(np.median(sub)), "med_dn": float(np.median(subn)),
              "M_real": float(np.median([real_mad[m] for m in keep])), "lopo_flip": []}
        if decide(s2)[0].split("_FRAGILE")[0].split("_CONFOUND")[0] != \
                v0.split("_FRAGILE")[0].split("_CONFOUND")[0]:
            st["lopo_flip"].append(p)

    verdict, why = decide(st)

    res = {"round": "M57", "verdict": verdict, "stats": st, "why": why,
           "gates": {"frac_neg": FRAC_NEG_GATE, "p": P_GATE, "mag": MAG_GATE,
                     "min_paired": MIN_PAIRED, "ruler_trivial": RULER_TRIVIAL_EXACT,
                     "copy_leak": COPY_LEAK_GATE, "pack_min_mats": PACK_MIN_MATS},
           "ruler16_on_our_gen": ruler16,
           "gen32_exact_rate": float(np.mean([readout(x)[0] for xs in g32.values() for x in xs]))
           if g32 else None,
           "real_side": {"n_tiles": len(real), "n_materials": len(real_by_mat),
                         "n_packs": len(set(pack_of_mat.values())),
                         "mad_median_pooled": float(np.median(
                             [readout(s["palette"][s["idx"]])[1] for s in real])),
                         "exact_rate_pooled": float(np.mean(
                             [readout(s["palette"][s["idx"]])[0] for s in real]))},
           "gen_side": {"n_32": n32, "n_16": n16, "n_materials": len(gen_mad),
                        "mad_median_pooled": float(np.median(list(gen_mad.values())))
                        if gen_mad else None},
           "copy_context": {"median_min_mad_to_same_material_real":
                            float(np.median(min_mad_to_real)) if min_mad_to_real else None,
                            "exact_copies": copy_hits, "gen32_counted": gen32_total},
           "by_pack": rows,
           "per_material": [{"material": m, "pack": pack_of_mat[m],
                             "mad_gen": gen_mad[m], "mad_real": real_mad[m],
                             "d": d_by_mat[m], "dn": dn_by_mat[m]} for m in paired]}

    if a.out:                      # 先落盘再打印（崩了数据也在）
        a.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    print(f"真人侧 base 32px 训练池：{res['real_side']['n_tiles']} 张 / "
          f"{res['real_side']['n_materials']} 材质 / {res['real_side']['n_packs']} 包；"
          f"mad 中位 {res['real_side']['mad_median_pooled']:.2f}；"
          f"全同 {res['real_side']['exact_rate_pooled']:.1%}")
    print(f"生成侧：32px {n32} 张、16px {n16} 张；配对材质 {st['n_paired']}；缺图 {st['n_missing_gen']}")
    print(f"(作废3) 16px 产物 vs 8px 平铺两遍：exact {ruler16['exact_rate']:.1%}"
          f"（门 {RULER_TRIVIAL_EXACT:.0%}）  mad 中位 "
          f"{ruler16['mad_median'] if ruler16['mad_median'] is None else round(ruler16['mad_median'], 2)}")
    print(f"(作废4) 逐像素抄回同材质真人瓦片：{copy_hits}/{gen32_total} = "
          f"{st['copy_leak_rate']:.1%}（门 {COPY_LEAK_GATE:.0%}）；"
          f"到同材质真人瓦片的最小 mad 中位 "
          f"{res['copy_context']['median_min_mad_to_same_material_real']}")
    print(f"主读数：M_real {st['M_real']:.3f}  med_d {st['med_d']:.3f} "
          f"（幅度门 {-MAG_GATE * st['M_real']:.3f}）  frac_neg {st['frac_neg']:.1%}"
          f"（门 {FRAC_NEG_GATE:.0%}）  符号检验 p {st['p_sign']:.3g}（门 {P_GATE}）")
    print(f"标准化副读数 med_dn {st['med_dn']:.4f}；三门：{why}")
    print(f"LOPO（材质数>={PACK_MIN_MATS} 的包 {len(big)} 个）翻侧："
          f"{st['lopo_flip'] if st['lopo_flip'] else '无'}")
    print("逐包（材质数前 6）")
    for r in rows[:6]:
        fn = "--" if r["frac_neg"] is None else f"{r['frac_neg']:.1%}"
        print(f"  {str(r['pack'])[:40]:<40} 材质 {r['n_mats']:>4}  med_d {r['med_d']:>8.3f}  d<0 {fn}")
    print(f"\n【判决】{verdict}")
    if a.out:
        print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
