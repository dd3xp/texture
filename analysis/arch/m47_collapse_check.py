"""(M47) 事后诊断：为什么「逐张取最近调色板」的 oracle 会让 KID 变**差**。

⛔ 这不是判决、不产生任何读数结论 —— 判读器已判 `VOID_METRIC_DISAGREE`，本脚本只回答
"那把量具坏在哪"，好决定下一轮要不要修它。

假说：逐张 argmin 是**多对一**的 —— 125 个目标会塌到少数几张"平均"调色板上，
分布多样性崩掉，于是逐张更近、整体更远。而 `palette=real` 是**一一对应**到参照集自己的调色板，
天然保分布，所以它才配当上界。

量：候选集越大，被选中的**去重调色板张数**越少 ⇒ 塌缩。零 GPU、零模型。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "eval"))
from tiles_data import load, canonicalise      # noqa: E402
from colour_task import targets                # noqa: E402


def pal_rs(p, n=16):
    p = np.asarray(p, np.float64)
    return p[np.round(np.linspace(0, len(p) - 1, n)).astype(np.int64)]


def main():
    bank = [np.asarray(s["palette"], np.uint8) for n in (16, 32) for s in load(n, "train", extra=True)]
    bank_rs = np.stack([pal_rs(p) for p in bank])
    T = targets("V_mat", 16)
    real = []
    for t in T:
        cols, inv = np.unique(t["ref"].reshape(-1, 3), axis=0, return_inverse=True)
        _, p = canonicalise(inv.reshape(16, 16).astype(np.int64), cols.astype(np.uint8))
        real.append(p)
    print(f"记忆库 {len(bank)} 张，V_mat 目标 {len(T)} 个")
    picks = []
    for rp in real:
        d = np.sqrt(((bank_rs - pal_rs(rp)) ** 2).sum(-1)).mean(-1)
        picks.append(int(d.argmin()))
    uniq = len(set(picks))
    print(f"全库 argmin：选中 {len(picks)} 次，去重 **{uniq}** 张（{uniq / len(picks):.1%}）")
    # 参照：真人 oracle 那一行天然一一对应
    print(f"palette=real 一行：去重 {len({tuple(p.reshape(-1)) for p in real})} 张（按构造一一对应）")
    # 被选中的调色板的"色彩张力"：平均饱和度跨度，塌缩时应明显低于真人
    def spread(p):
        p = np.asarray(p, np.float64)
        return float(p.max(0).mean() - p.min(0).mean())
    print(f"平均亮度跨度  真人 {np.mean([spread(p) for p in real]):.1f}  "
          f"全库 argmin {np.mean([spread(bank[i]) for i in picks]):.1f}")


if __name__ == "__main__":
    main()
