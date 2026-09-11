"""补 32px 训练数据：训练包里的 64px 真人瓦片按 2×2 块取众数降到 32px（v11）。

32px 是当前短板（验证集判官 28–36%，生成的是满屏单像素噪点）：32px 训练瓦片只有 641 张，
而训练包里有 1238 张 64px 真人瓦片。取众数（不取平均）保住像素画的清晰色块与原调色板，
不引入新颜色；降采样后没用到的颜色丢掉。只用训练包（与 dataset_k16 / 补回数据同一口径，已去污染）。

输出 `data/tiles/train_64to32.json`（样本格式同 dataset_k16，size = 32，pack 不变）。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
from tiles_data import load   # noqa: E402


def mode_pool2(idx):
    """[64,64] 索引 → [32,32]：每个 2×2 块取出现最多的索引（并列取亮度序更暗的，结果确定）。"""
    n = idx.shape[0] // 2
    blocks = idx.reshape(n, 2, n, 2).transpose(0, 2, 1, 3).reshape(n, n, 4)
    k = int(idx.max()) + 1
    counts = (blocks[..., None] == np.arange(k)).sum(2)          # [n,n,k]
    return counts.argmax(-1)


def main():
    out = []
    for s in load(64, "train", extra="train_extra_packs_only.json"):
        g = mode_pool2(np.asarray(s["idx"]))
        used = np.unique(g)
        if len(used) < 3:
            continue
        remap = np.full(int(g.max()) + 1, -1)
        remap[used] = np.arange(len(used))
        g = remap[g]
        pal = np.asarray(s["palette"], np.uint8)[used]            # 仍按亮度升序（used 升序、原调色板已亮度排序）
        out.append({"material": s["material"], "pack": s["pack"], "size": 32, "split": "train",
                    "native_colors": int(len(used)), "k_used": int(len(used)),
                    "idx": g.astype(np.uint8).tobytes().hex(), "palette": pal.tolist(), "extra": True})
    p = ROOT / "data/tiles/train_64to32.json"
    p.write_text(json.dumps({"k": 16, "samples": out}))
    print(f"64→32 样本 {len(out)} -> {p}")


if __name__ == "__main__":
    main()
