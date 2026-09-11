"""检索增强调色板：真人调色板记忆库。

动机（`eval/diag_decompose.py`，验证集）：TRD 与真人的分布差距**几乎全在配色**——
TRD 网格配真人调色板 KID 4.5（真人对真人地板 3.8），真人网格配 TRD 调色板 KID 24.1（TRD 原样 27.9）。
结构已经到了画师水平，调色板没有。画师的调色板是少量带色相偏移的明度阶梯，4000 张训练瓦片
不够让 512 码的离散头学好这种联合分布；但这些调色板本身就在训练集里。

做法（与 Retrieval-Augmented Diffusion / kNN-Diffusion 同一思路，只检索配色、不检索结构）：
- 记忆库 = 训练集（去污染 + 补回，16/32px）每张瓦片的 (材质名 CLIP 文本嵌入, 色数 k, 亮度序调色板, 平均色)。
- 查询 (文本嵌入, k[, 区域颜色])：同 k 的条目里取文本相似度前 n_text，
  给了区域颜色时再按平均色 ΔE 排，取前 topk 随机一个。
- 区域颜色任务：检索到的调色板在 Lab 里整体平移，使那张瓦片的平均色落到目标色上。
- 生成：检索到的调色板作为**已知 token** 喂给 TRD（训练时的随机掩码本就包含"调色板已知、网格被掩"），
  TRD 只生成网格——结构是新生成的，不是拷贝的。
"""
import numpy as np
import torch

from tiles_data import load


def _lab(rgb):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    from colour_task import rgb_to_lab
    return rgb_to_lab(rgb)


class PaletteMemory:
    def __init__(self, dev, sizes=(16, 32)):
        from train_trd import clip_text, text_prompt
        rows = [s for n in sizes for s in load(n, "train", extra=True)]
        self.pal = [np.asarray(s["palette"], np.uint8) for s in rows]
        self.k = np.array([s["k_used"] for s in rows])
        self.hist = [np.bincount(s["idx"].reshape(-1), minlength=s["k_used"])[:s["k_used"]] / s["idx"].size
                     for s in rows]                    # 各秩的像素占比（离散 SDEdit 做直方图匹配用）
        self.mean = np.stack([s["palette"][s["idx"]].reshape(-1, 3).mean(0) for s in rows])
        self.mean_lab = _lab(self.mean)
        mats = sorted({s["material"] for s in rows})
        emb = clip_text([text_prompt(m) for m in mats], dev).float()
        ix = {m: i for i, m in enumerate(mats)}
        self.emb = emb[torch.tensor([ix[s["material"]] for s in rows])].to(dev)     # [M, D]
        self.material = [s["material"] for s in rows]

    def query(self, text_emb, k, rng, colour=None, topk=5, n_text=50):
        """返回 (调色板 uint8 [k,3], 条目下标)。k 不存在时退到最近的 k。"""
        ks = np.unique(self.k)
        kk = k if k in ks else ks[np.abs(ks - k).argmin()]
        pool = np.nonzero(self.k == kk)[0]
        sims = (self.emb[torch.as_tensor(pool, device=self.emb.device)] @ text_emb.float()).cpu().numpy()
        cand = pool[np.argsort(-sims)[:n_text if colour is not None else topk]]
        if colour is not None:
            de = np.linalg.norm(self.mean_lab[cand] - _lab(np.asarray(colour, float)), axis=1)
            cand = cand[np.argsort(de)[:topk]]
        i = int(cand[rng.integers(len(cand))])
        return self.pal[i], i

    def shifted(self, i, colour):
        """把第 i 条的调色板在 Lab 里平移，使该瓦片平均色落到 colour 上。"""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
        from colour_task import lab_to_rgb
        d = _lab(np.asarray(colour, float)) - self.mean_lab[i]
        return lab_to_rgb(_lab(self.pal[i]) + d)
