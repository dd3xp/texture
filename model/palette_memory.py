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
from pathlib import Path

import numpy as np
import torch

from tiles_data import load


def _lab(rgb):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    from colour_task import rgb_to_lab
    return rgb_to_lab(rgb)


CLIP16 = Path(__file__).resolve().parents[1] / "weights/clip-vit-base-patch16"


@torch.no_grad()
def clip16(dev):
    from transformers import CLIPModel, CLIPTokenizer
    return CLIPModel.from_pretrained(str(CLIP16)).to(dev).eval(), CLIPTokenizer.from_pretrained(str(CLIP16))


@torch.no_grad()
def clip16_images(tiles, dev, model=None, bs=256):
    """瓦片（任意边长）最近邻放大到 224 → CLIP-B/16 图像嵌入（归一化）。跨模态检索用；评测用的是 B/32。"""
    import torch.nn.functional as F
    m = model or clip16(dev)[0]
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(tiles), bs):
        x = torch.cat([F.interpolate(torch.from_numpy(np.asarray(t)).permute(2, 0, 1)[None].float().to(dev),
                                     size=224, mode="nearest") for t in tiles[i:i + bs]]) / 255.0
        out.append(F.normalize(m.get_image_features(pixel_values=(x - mean) / std).float(), dim=-1))
    return torch.cat(out)


@torch.no_grad()
def clip16_texts(prompts, dev, model=None, tok=None):
    import torch.nn.functional as F
    if model is None:
        model, tok = clip16(dev)
    t = tok([f"pixel art texture of {p}" for p in prompts], padding=True, return_tensors="pt").to(dev)
    return F.normalize(model.get_text_features(**t).float(), dim=-1)


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
        self.tiles = [s["palette"][s["idx"]] for s in rows]
        self.img16 = None

    def enable_xmodal(self, dev):
        """跨模态检索：给记忆库每张真人瓦片算 CLIP-B/16 图像嵌入（query 时按"瓦片 ↔ 材质名"的图文相似度重排）。"""
        self.img16 = clip16_images(self.tiles, dev)

    def query(self, text_emb, k, rng, colour=None, topk=5, n_text=50, text16=None, n_name=30,
              xpal_temp=None):
        """返回 (调色板 uint8 [k,3], 条目下标)。k=None：不限色数，色数跟检索到的调色板走（推荐）；
        给 k 时只在恰好 k 色的条目里找（k 不存在时退到最近的 k）。

        为什么推荐 None：先从全局分布里抽 k 再限定 k 色，训练集里该材质若没有这个色数的版本，
        就会退到名字相近但颜色完全不同的材质（验证集判官输掉的对里："coal block" 绿、"snow" 深灰、"diamond block" 黄绿）。"""
        if k is None:
            pool = np.arange(len(self.k))
        else:
            ks = np.unique(self.k)
            kk = k if k in ks else ks[np.abs(ks - k).argmin()]
            pool = np.nonzero(self.k == kk)[0]
        sims = (self.emb[torch.as_tensor(pool, device=self.emb.device)] @ text_emb.float()).cpu().numpy()
        sims = sims + 1e-6 * rng.random(len(sims))       # 同名材质相似度完全相同：随机打破平局，否则总取同样几张（多样性掉）
        if colour is None and text16 is not None and self.img16 is not None:
            # 跨模态：先按名字取 n_name 个同/近名条目，再按"这张真人瓦片 ↔ 材质名"的 B/16 图文相似度取前 topk
            # （同名材质里挑画得最典型的那几张的配色；KNN-Diffusion 的检索方式）
            near = pool[np.argsort(-sims)[:n_name]]
            xs = (self.img16[torch.as_tensor(near, device=self.img16.device)] @ text16.float()).cpu().numpy()
            if xpal_temp:
                # 软典型度：在 n_name 个候选上按 softmax(图文相似度 / 温度) 抽一个，而不是硬取前 topk。
                # 温度 →0 等于取最典型的一张；温度很大等于名字检索（不看典型度）。
                w = np.exp((xs - xs.max()) / xpal_temp)
                i = int(near[rng.choice(len(near), p=w / w.sum())])
            else:
                cand = near[np.argsort(-xs)[:topk]]
                i = int(cand[rng.integers(len(cand))])
            return self.pal[i], i
        cand = pool[np.argsort(-sims)[:n_text if colour is not None else topk]]
        if colour is not None:
            de = np.linalg.norm(self.mean_lab[cand] - _lab(np.asarray(colour, float)), axis=1)
            if text16 is not None and self.img16 is not None:
                # 区域颜色 + 跨模态：先按颜色留最近的 3×topk 个，再按"瓦片 ↔ 材质名"的 B/16 相似度取前 topk
                near = cand[np.argsort(de)[:3 * topk]]
                xs = (self.img16[torch.as_tensor(near, device=self.img16.device)] @ text16.float()).cpu().numpy()
                cand = near[np.argsort(-xs)[:topk]]
            else:
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
