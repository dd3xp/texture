"""613bdbb 那个"真人 32px"参照组是什么？——零 GPU、零 API、净克隆可跑。

**为什么要有这个脚本**：`analysis/arch/scale_diag.py`（613bdbb）把 32px 的缺口写成
"TRD 各向异性 0.130 / 过门 28%，**真人** 0.534 / 64%"，后续六条杠杆全是照这个差距在挑配置。
`train_pool_structure.py` 的事后一段发现：那个"真人"参照组（val32 与 `V_mat` 的交集）
**只有 74 张，其中 66 张来自同一个包 `ROllerozxa__mtg_tiled_32x`**（包名里就写着 tiled）。
于是"真人 32px"这个词指的其实是**一个高度平铺的包**，不是语料库里的真人 32px。

本脚本把这件事钉成可复算的数：参照组的来源包 + 它对**每一个**已量过的组的对比。
参照组这一侧只用已入库数据（`data/tiles/dataset_k16.json` + `eval/prompt_sets_val.json`），
训练池那一侧读已入库的 `experiments/train_pool_structure.json`（池子的 extra 文件只在服务器上，
所以那几组不能在本机重量——但它们的逐张各向异性已经落盘）。

    python analysis/arch/ref_pack_audit.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "analysis/arch"))
from downsample import dominant_period, anisotropy              # noqa: E402
from tiles_data import load                                     # noqa: E402
from train_pool_structure import mannwhitney_u_p, two_prop      # noqa: E402

GATE_ANISO = 0.20          # 与 scale_diag.py / train_pool_structure.py 同口径
POOLS = ROOT / "experiments/train_pool_structure.json"


def reference_group():
    """613bdbb 用的参照组：val32 里落在 V_mat 的那些材质。"""
    vmat = {p["material"] for p in
            json.loads((ROOT / "eval/prompt_sets_val.json").read_text(encoding="utf-8"))["V_mat"]}
    val32 = load(32, "val")
    ref = [s for s in val32 if s["material"] in vmat]
    return ref, val32


def profile(rows):
    anis, gated = [], 0
    for s in rows:
        rgb = s["palette"][s["idx"]].astype(float)
        a = anisotropy(rgb)
        anis.append(a)
        gated += bool(dominant_period(rgb, lo=2, hi_frac=0.625) > 0 and a >= GATE_ANISO)
    return {"n": len(rows), "n_gated": gated, "gate_rate": gated / len(rows),
            "median_aniso": float(np.median(anis)), "anisos": anis}


def main():
    ref, val32 = reference_group()
    packs = Counter(s["pack"] for s in ref)
    top, ntop = packs.most_common(1)[0]
    R = profile(ref)

    print("613bdbb 的『真人 32px』参照组")
    print(f"  n={R['n']}  过门 {R['gate_rate']:.0%}  aniso 中位 {R['median_aniso']:.3f}")
    print(f"  来源包 {dict(packs)}")
    print(f"  => 最大的一个包 {top} 占 {ntop}/{R['n']} = {ntop / R['n']:.0%}")
    assert R["n"] > 0, "参照组为空：prompt_sets_val.json 与 dataset_k16.json 对不上"

    rows = {"整份 val32（真人，未按 V_mat 过滤）": profile(val32)}
    if POOLS.exists():
        for k, v in json.loads(POOLS.read_text(encoding="utf-8"))["groups"].items():
            if not k.startswith("C  "):        # C 就是参照前的整份 val32，已单列
                rows[k] = v
    else:
        print(f"\n[跳过] 缺 {POOLS}，只报参照组本身")

    print("\n每一组 vs 参照组（各向异性 Mann-Whitney 双侧；过门率两比例）")
    print(f"  {'组':<32}{'n':>6}{'过门':>7}{'aniso':>8}{'MW p':>12}{'门 p':>12}")
    out = {}
    for k, g in rows.items():
        _, pa = mannwhitney_u_p(g["anisos"], R["anisos"])
        pg = two_prop(g["n_gated"], g["n"], R["n_gated"], R["n"])
        out[k] = {"n": g["n"], "gate_rate": g["gate_rate"],
                  "median_aniso": g["median_aniso"], "aniso_p": pa, "gate_p": pg}
        print(f"  {k:<32}{g['n']:>6}{g['gate_rate']:>7.0%}"
              f"{g['median_aniso']:>8.3f}{pa:>12.3g}{pg:>12.3g}")

    print("\n读法：参照组比语料库里**任何**一组都更有结构——包括真人自己的 16px 验证集。")
    print("      TRD 32px 量到的 0.130 / 28%（613bdbb）与整份 val32 的 "
          f"{rows['整份 val32（真人，未按 V_mat 过滤）']['median_aniso']:.3f} / "
          f"{rows['整份 val32（真人，未按 V_mat 过滤）']['gate_rate']:.0%} 是一回事。")
    print("      所以『32px 上真人有结构而 TRD 没有』的差距，量的是**一个平铺包**，不是真人 32px。")
    print("      判官那条（32px 输，41%，p=0.014）是另一把尺子，**不受本脚本影响**。")
    return out


if __name__ == "__main__":
    main()
