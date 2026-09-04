"""配对翻译可不可学？训练之前先回答。

A4 证明降采样基线在人的偏好上强于我们的模型（83:17），
且基线的结构来自 SDXL 这样的基础模型，而我们从零训练、每类约 5 个样本。
所以把任务改成**有监督配对翻译**：基线输出 -> 真人瓦片。

但改形式之前必须先问：**基线与真人的差异是系统性的，还是作者的任意选择？**
若是后者，任何模型都学不出来，这条路当场就该停。

检验（不训练网络）：
  1. 原样照抄（identity）的逐格准确率——基线本身有多接近真人
  2. 只用一个**全局索引重映射**（从训练集统计 P(真人档位 | 基线档位)）能否胜过照抄
  3. 加上"局部对比度"这一维条件后能否再涨

任何一步涨了，就说明差异里有可学的规律。都不涨就停。
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "metric"))
from build_testset import match_stats                              # noqa: E402


def baseline_idx(src_path: str, art_rgb: np.ndarray, pal: np.ndarray,
                 size: int = 16):
    """高分源 -> 降采样 -> 色彩对齐 -> 量化到该瓦片调色板。与 A4 用的基线一致。"""
    try:
        tex = Image.open(src_path).convert("RGB")
    except Exception:
        return None
    x = np.asarray(tex.resize((size, size), Image.BOX), float)
    x = match_stats(x, art_rgb.astype(float))
    d = ((x.reshape(-1, 1, 3) - pal.astype(float).reshape(1, -1, 3)) ** 2).sum(-1)
    return d.argmin(1).reshape(size, size)


def main():
    ds = json.loads(Path("data/tiles/dataset_k16.json").read_text())
    pairs = json.loads(Path("data/contentdb/pairs.json").read_text())
    s16 = [s for s in ds["samples"] if s["size"] == 16]

    data = {"train": [], "test": []}
    n_mat = 0
    for s in s16:
        m = s["material"]
        if m not in pairs or s["split"] == "val":
            continue
        pal = np.array(s["palette"], np.uint8)
        art = np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16)
        art_rgb = pal[art]
        for src in pairs[m].get("high", {}).values():
            b = baseline_idx(src, art_rgb, pal)
            if b is None:
                continue
            data[s["split"]].append((b.astype(np.int64), art.astype(np.int64),
                                     len(pal), m))
    n_mat = len({d[3] for d in data["train"]})
    print(f"配对样本：训练 {len(data['train'])}，测试 {len(data['test'])}，"
          f"材质 {n_mat}", flush=True)
    if not data["train"]:
        print("没有配对样本，停")
        return

    def acc(pred_fn, split):
        ok = tot = 0
        for b, a, nk, m in data[split]:
            p = pred_fn(b, nk, m)
            ok += int((p == a).sum())
            tot += a.size
        return ok / max(tot, 1)

    # 1) 原样照抄
    ident = lambda b, nk, m: b
    a_id_tr, a_id_te = acc(ident, "train"), acc(ident, "test")
    print(f"\n1) 原样照抄        训练 {a_id_tr:.3f}   测试 {a_id_te:.3f}")

    # 2) 全局索引重映射：把档位归一到 [0,1] 再分 16 档统计
    B = 16
    cnt = np.zeros((B, B))
    for b, a, nk, m in data["train"]:
        bb = np.clip((b / max(nk - 1, 1) * (B - 1)).round().astype(int), 0, B - 1)
        aa = np.clip((a / max(nk - 1, 1) * (B - 1)).round().astype(int), 0, B - 1)
        np.add.at(cnt, (bb.ravel(), aa.ravel()), 1)
    remap = cnt.argmax(1)

    def f_remap(b, nk, m):
        bb = np.clip((b / max(nk - 1, 1) * (B - 1)).round().astype(int), 0, B - 1)
        out = remap[bb] / (B - 1) * max(nk - 1, 1)
        return np.clip(out.round().astype(int), 0, nk - 1)
    print(f"2) 全局重映射      训练 {acc(f_remap,'train'):.3f}   "
          f"测试 {acc(f_remap,'test'):.3f}")

    # 3) 加一维局部条件：该格与 3x3 邻域均值的差（分 5 档）
    def ctx(b, nk):
        pad = np.pad(b.astype(float), 1, mode="edge")
        loc = np.stack([pad[i:i+16, j:j+16] for i in range(3) for j in range(3)])
        rel = (b - loc.mean(0)) / max(nk - 1, 1)
        return np.clip(((rel + 0.5) * 4).round().astype(int), 0, 4)

    cnt3 = np.zeros((B, 5, B))
    for b, a, nk, m in data["train"]:
        bb = np.clip((b / max(nk-1,1) * (B-1)).round().astype(int), 0, B-1)
        aa = np.clip((a / max(nk-1,1) * (B-1)).round().astype(int), 0, B-1)
        cc = ctx(b, nk)
        np.add.at(cnt3, (bb.ravel(), cc.ravel(), aa.ravel()), 1)
    remap3 = cnt3.argmax(2)

    def f_ctx(b, nk, m):
        bb = np.clip((b / max(nk-1,1) * (B-1)).round().astype(int), 0, B-1)
        out = remap3[bb, ctx(b, nk)] / (B - 1) * max(nk - 1, 1)
        return np.clip(out.round().astype(int), 0, nk - 1)
    print(f"3) 重映射+局部条件 训练 {acc(f_ctx,'train'):.3f}   "
          f"测试 {acc(f_ctx,'test'):.3f}")

    print("\n判读：测试准确率若能明显高于「原样照抄」，说明差异有系统规律，可学；")
    print("      三者接近 -> 差异是作者的任意选择，配对翻译这条路也该停。")


if __name__ == "__main__":
    main()
