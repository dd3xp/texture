"""训练数据：把 D2 产出的索引图喂给模型，带二面体群增广与调色板扰动。

增广用 4 旋转 × 2 翻转（二面体群 D4）。这对材质是合法的——
木纹旋转 90 度仍是木纹，材质近似各向同性。
对**有方向性**的材质（比如草的上边缘）这条不严格成立，
但那类瓦片在数据集里已被 `_top$`/`_side$` 之类的规则过滤掉大半。
5979 张 × 8 ≈ 4.8 万有效样本。

**调色板扰动**：同一张索引图配不同调色板。这既扩数据，又直接贴合任务——
模型本该学"给定任意调色板去填"，而不是记住"木头就是那几个棕色"。
扰动在 HSV 上做：整体色相平移、饱和度缩放、逐档明度抖动，
之后**按亮度重排并重映射索引**，保证"相邻索引 = 相邻明度"这个约定不被破坏。

**没有做**把 32/64 下采样到 16 来补数据——下采样正是本路线要避免的平均运算，
那样生成的样本会把中间色的模式教给模型，等于往训练集里投毒。

**做了**从 32/64 瓦片**随机裁剪**出 16×16 块（`crop_larger=True`）。
裁剪不是平均：每个像素都是艺术家画的原始像素，没有任何插值。
训练集从 3395 张扩到 3395+853=4248 个样本源，
每次取样裁不同位置，等效数据量约 7 倍（stride 8 计）。

**隐患（需实测）**：32×32 裁出的 16×16 块，材质的像素尺度是原生 16×16 的两倍，
砖块看起来大一倍。这可能是有用的尺度增广，也可能是干扰。
`--crop-larger` 开关就是为了做这个对照。
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TileSet(Dataset):
    """索引图 + 调色板 + 材质 id。"""

    def __init__(self, path: Path, split: str, size: int = 16, augment: bool = True,
                 palette_jitter: float = 0.0, crop_larger: bool = False):
        blob = json.loads(Path(path).read_text())
        self.k = blob["k"]
        self.augment = augment
        self.palette_jitter = palette_jitter
        self.size = size
        self.crop_larger = crop_larger
        self.samples = [s for s in blob["samples"]
                        if s["split"] == split
                        and (s["size"] == size or
                             (crop_larger and s["size"] > size))]
        mats = sorted({s["material"] for s in blob["samples"]})
        self.mat2id = {m: i for i, m in enumerate(mats)}
        self.n_materials = len(mats)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        s = self.samples[i]
        n = s["size"]
        idx = np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(n, n).copy()

        if n > self.size:
            # 随机裁剪。每次取样裁不同位置，等效于无限多块。
            # 注意这不是下采样——像素本身没被平均，只是取了一个窗口。
            m = self.size
            y = np.random.randint(0, n - m + 1)
            x = np.random.randint(0, n - m + 1)
            idx = idx[y:y + m, x:x + m].copy()

        if self.augment:
            r = np.random.randint(4)
            if r:
                idx = np.rot90(idx, r).copy()
            if np.random.rand() < 0.5:
                idx = np.fliplr(idx).copy()

        p = np.array(s["palette"], np.float32) / 255.0
        if self.augment and self.palette_jitter > 0:
            p, idx = _jitter_palette(p, idx, self.palette_jitter)

        pal = np.zeros((self.k, 3), np.float32)
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


def _jitter_palette(p: np.ndarray, idx: np.ndarray, amount: float):
    """在 HSV 上扰动调色板，然后按亮度重排并同步重映射索引。

    重排是必须的：索引的语义是"按亮度排第几档"，
    扰动后若不重排，这个约定就被破坏了，模型学到的"跳几档"会失去意义。
    """
    import colorsys

    hue_shift = (np.random.rand() - 0.5) * 2 * amount          # 全局同移，保持色彩关系
    sat_scale = 1.0 + (np.random.rand() - 0.5) * 2 * amount
    out = np.empty_like(p)
    for i, (r, g, b) in enumerate(p):
        h, s_, v = colorsys.rgb_to_hsv(float(r), float(g), float(b))
        h = (h + hue_shift) % 1.0
        s_ = float(np.clip(s_ * sat_scale, 0.0, 1.0))
        v = float(np.clip(v + (np.random.rand() - 0.5) * amount, 0.0, 1.0))
        out[i] = colorsys.hsv_to_rgb(h, s_, v)

    lum = out @ np.array([0.299, 0.587, 0.114], np.float32)
    order = np.argsort(lum)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return out[order], inv[idx]
