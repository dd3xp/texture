"""训练数据：把 D2 产出的索引图喂给模型，带二面体群增广。

增广用 4 旋转 × 2 翻转（二面体群 D4）。这对材质是合法的——
木纹旋转 90 度仍是木纹，材质近似各向同性。
对**有方向性**的材质（比如草的上边缘）这条不严格成立，
但那类瓦片在数据集里已被 `_top$`/`_side$` 之类的规则过滤掉大半。
5979 张 × 8 ≈ 4.8 万有效样本。
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TileSet(Dataset):
    """索引图 + 调色板 + 材质 id。"""

    def __init__(self, path: Path, split: str, size: int = 16, augment: bool = True):
        blob = json.loads(Path(path).read_text())
        self.k = blob["k"]
        self.augment = augment
        self.size = size
        self.samples = [s for s in blob["samples"]
                        if s["split"] == split and s["size"] == size]
        mats = sorted({s["material"] for s in blob["samples"]})
        self.mat2id = {m: i for i, m in enumerate(mats)}
        self.n_materials = len(mats)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        s = self.samples[i]
        n = s["size"]
        idx = np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(n, n).copy()

        if self.augment:
            r = np.random.randint(4)
            if r:
                idx = np.rot90(idx, r).copy()
            if np.random.rand() < 0.5:
                idx = np.fliplr(idx).copy()

        pal = np.zeros((self.k, 3), np.float32)
        p = np.array(s["palette"], np.float32) / 255.0
        pal[: len(p)] = p
        # 调色板有效位：实际用了几档。不足 K 的部分要被模型忽略
        valid = np.zeros(self.k, np.float32)
        valid[: len(p)] = 1.0

        return {
            "idx": torch.from_numpy(idx.astype(np.int64)),
            "palette": torch.from_numpy(pal),
            "pal_valid": torch.from_numpy(valid),
            "material": torch.tensor(self.mat2id[s["material"]], dtype=torch.long),
            "n_used": torch.tensor(s["k_used"], dtype=torch.long),
        }
