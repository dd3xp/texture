"""逐材质决定要不要对横缝设门。

A3y 用验收过的尺子回测发现：门一刀切不成立。
被它拦下的 9 个材质里，`mcl_core_gold_ore` 大幅改善（125.77→42.16），
而 fence 三兄弟与 `default_acacia_wood` 明显变差（如 0.10→0.77）——
栅栏本来就有真实横向结构，**置换检验把它们误杀了**。

所以照 `bond_select.py` 的模式逐材质选。两种配置的瓦片存档都在，不必重跑。

**不用测试瓦片选**：结构尺子本身就是"到该材质训练瓦片描述子分布的距离"，
直接取分数低的那个配置即可，不碰测试集也不需要人工标注。
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from struct_metric import make_scorer                              # noqa: E402


def hx(h):
    return np.frombuffer(bytes.fromhex(h), np.uint8).reshape(16, 16)


def main():
    A = {r["material"]: r for r in json.loads(
        Path("experiments/struct_metric_tiles_preA3v.json").read_text())}
    B = {r["material"]: r for r in json.loads(
        Path("experiments/struct_metric_tiles_full.json").read_text())}

    out, n_gate, n_keep = {}, 0, 0
    print(f"{'材质':<34}{'不设门':>9}{'设门':>9}{'选用':>8}")
    print("-" * 62)
    for m in sorted(set(A) & set(B)):
        a, b = A[m], B[m]
        # 只有两种配置的种子确实不同才需要选
        if a["seeded"] == b["seeded"]:
            continue
        bands = [tuple(x) for x in a["bands"]]
        sc = make_scorer([hx(h) for h in a["train"]], bands)
        sa = float(np.median([sc(hx(h)) for h in a["seeded"]]))
        sb = float(np.median([sc(hx(h)) for h in b["seeded"]]))
        gate = sb < sa
        out[m] = {"gate": bool(gate), "no_gate_score": round(sa, 4),
                  "gate_score": round(sb, 4)}
        n_gate += gate
        n_keep += not gate
        print(f"{m:<34}{sa:>9.3f}{sb:>9.3f}"
              f"{('设门' if gate else '不设门'):>10}")
    f = Path("data/tiles/seam_gate_select.json")
    f.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\n受影响材质 {len(out)}：选设门 {n_gate}，选不设门 {n_keep}")
    print(f"写入 {f}")
    print("注意：只用训练瓦片的描述子分布打分，不碰测试集。")


if __name__ == "__main__":
    main()
