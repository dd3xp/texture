"""原生 16x16 离散瓦片数据层：把真人瓦片变成**亮度规范化的索引网格**。

**为什么是这条路**（与已关闭的结论逐条对齐）：
- 「没有标准答案」（两个真人逐格一致率 9.8%，配对零假设 8.6%）→
  **逐像素重建损失是错的目标**，A4 掩码预测就死在这（人工盲比 17:83 输给朴素降采样）。
  正确的目标是**建模分布**，不是拟合唯一答案。
- LoRA 微调走的是分布式目标（扩散损失），方向对，但它在
  **1024 放大图 + SDXL 的 VAE** 上做——隔着两层去改一个 16x16 的性质，两次都失败。
- 所以还没走过、且与上述都不矛盾的是：**在 16x16 索引空间里直接建模**。
  结构单元惯例（真人每图约 3.2 个）是这个空间里的一阶统计量，
  原生建模的话它**从数据里自然来**，不需要裁剪那种推理期启发式。

**关键设计：亮度规范化**。每张瓦片的调色板索引语义不同（0 号在这张是深棕、
在那张是白）。不规范化的话「索引 k」没有跨样本含义，模型学不到结构。
这里把每张的调色板**按亮度升序重排并重新编号**，于是索引 k 恒定表示
「本图第 k 暗的颜色」——结构与配色就此解耦，正合任务本身
（纯色 + 材质名 → 纹理：颜色由用户给，结构由模型出）。

产出 `(idx[16,16] int64, k_used, words)`，其中 `words` 是材质名清洗后的词表。
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
W_LUM = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def clean_name(material: str) -> list[str]:
    """`baked_clay_black.png` -> ['baked','clay','black']。"""
    s = re.sub(r"\.(png|jpg|jpeg)$", "", material.lower())
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return [w for w in s.split() if w and not w.isdigit()]


def canonicalise(idx: np.ndarray, palette: np.ndarray):
    """按亮度升序重排调色板并重编号索引。返回 (新索引, 新调色板)。

    只重排**实际用到**的颜色：未出现的条目会让 k_used 虚高，
    而模型要学的是"这张图用了几个色阶、怎么排布"。
    """
    used = np.unique(idx)
    pal_used = palette[used]
    order = np.argsort(pal_used.astype(np.float64) @ W_LUM, kind="stable")
    remap = np.zeros(int(palette.shape[0]), dtype=np.int64)
    for new, old_pos in enumerate(order):
        remap[used[old_pos]] = new
    return remap[idx], pal_used[order]


def load(size: int = 16, split: str | None = None):
    """产出 dict(idx, palette, k_used, words, material, pack, split)。"""
    ds = json.loads((ROOT / "data/tiles/dataset_k16.json").read_text())
    out = []
    for s in ds["samples"]:
        if s["size"] != size:
            continue
        if split is not None and s.get("split") != split:
            continue
        n = s["size"]
        raw = np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(n, n).astype(np.int64)
        pal = np.array(s["palette"], dtype=np.uint8)
        cid, cpal = canonicalise(raw, pal)
        out.append({"idx": cid, "palette": cpal, "k_used": int(cpal.shape[0]),
                    "words": clean_name(s["material"]), "material": s["material"],
                    "pack": s.get("pack"), "split": s.get("split")})
    return out


def vocab(samples, min_count: int = 2) -> dict[str, int]:
    """材质词表。低频词丢给 <unk>，让未见材质名也能落到已知词上。"""
    from collections import Counter
    c = Counter(w for s in samples for w in s["words"])
    words = [w for w, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])) if n >= min_count]
    return {"<unk>": 0, **{w: i + 1 for i, w in enumerate(words)}}


def _selftest():
    """规范化必须**保持图像不变**：重排索引 + 重排调色板 = 同一张图。"""
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(200):
        k = int(rng.integers(2, 17))
        pal = rng.integers(0, 256, (k, 3)).astype(np.uint8)
        idx = rng.integers(0, k, (16, 16)).astype(np.int64)
        cid, cpal = canonicalise(idx, pal)
        if not np.array_equal(pal[idx], cpal[cid]):
            bad += 1
        lum = cpal.astype(np.float64) @ W_LUM
        if not np.all(np.diff(lum) >= -1e-9):
            bad += 1
    print(f"规范化自检 200 例：像素级不变 + 亮度单调  失败 {bad}")
    real = load(16)
    ok = sum(1 for s in real[:300]
             if s["idx"].max() < s["k_used"] and s["idx"].min() >= 0)
    print(f"真实数据 300 例：索引落在 [0,k_used) 的 {ok}/300")
    v = vocab(real)
    cov = np.mean([np.mean([w in v for w in s["words"]]) for s in real])
    print(f"样本 {len(real)}；词表 {len(v)}；材质词被覆盖率 {cov:.1%}")
    import collections
    print("k_used 分布（前6）:",
          collections.Counter(s["k_used"] for s in real).most_common(6))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
