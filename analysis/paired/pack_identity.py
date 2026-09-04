"""风格是否存在于单张瓦片里？——决定"包级一致性"是不是伪命题。

B1 量出不同作者画同一材质时逐格一致率低于随机（0.098 vs 0.172）。
一个解释是：**艺术家的决策不在单张瓦片上，在整套材质包上**，
两人画同一块砖不一致，是各自服从自己那套包的风格。

若成立，则单张瓦片里应当能认出"这是哪个包"。
检验：只用调色板与颗粒统计（不看具体像素排布），
按材质做留一划分，预测瓦片属于哪个包。

留一是关键：训练与测试**不共享材质**，
否则模型可以靠"记住这个材质长什么样"作弊，而不是学风格。
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np


def feats(idx: np.ndarray, pal: np.ndarray) -> np.ndarray:
    """只用风格性统计，不含材质特有的空间排布。"""
    W = np.array([0.299, 0.587, 0.114])
    lum = np.sort(pal.astype(float) @ W)
    a = idx.astype(float)
    nk = max(len(pal) - 1, 1)
    hsv_like = pal.astype(float) / 255.0
    sat = (hsv_like.max(1) - hsv_like.min(1)).mean()
    return np.array([
        len(pal),                                   # 色数
        lum[-1] - lum[0],                           # 亮度跨度
        float(np.median(np.diff(lum))) if len(lum) > 1 else 0.0,  # 档距中位
        lum.mean(), sat,                            # 平均亮度、平均饱和
        float((a[:, :-1] == a[:, 1:]).mean()),      # 相邻同色（平坦度）
        float(np.abs(np.diff(a, axis=1)).mean()) / nk,   # 横向跳变
        float(np.abs(np.diff(a, axis=0)).mean()) / nk,   # 纵向跳变
        float(a.std()) / nk,                        # 档位离散度
        float(len(np.unique(idx))) / max(len(pal), 1),   # 实际用色比例
    ])


def main():
    ds = json.loads(Path("data/tiles/dataset_k16.json").read_text())
    pairs = json.loads(Path("data/contentdb/pairs.json").read_text())
    # 材质 -> {包: 路径}，用于给瓦片打包标签
    mat2pack = {m: set(v.get("low", {})) for m, v in pairs.items()}

    X, y, mats = [], [], []
    for s in ds["samples"]:
        if s["size"] != 16:
            continue
        m, src = s["material"], s.get("source") or s.get("pack")
        if src is None:
            continue
        pal = np.array(s["palette"], np.uint8)
        idx = np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16)
        X.append(feats(idx, pal)); y.append(src); mats.append(m)
    if not X:
        print("样本里没有包标签字段，检查 dataset 的键：",
              list(ds["samples"][0].keys()))
        return
    X = np.array(X); y = np.array(y); mats = np.array(mats)
    cnt = Counter(y)
    keep = {k for k, v in cnt.items() if v >= 30}
    sel = np.array([v in keep for v in y])
    X, y, mats = X[sel], y[sel], mats[sel]
    print(f"瓦片 {len(X)}，包 {len(keep)}，材质 {len(set(mats))}")
    print(f"多数类基准 {Counter(y).most_common(1)[0][1]/len(y):.3f}")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)
    accs = []
    for tr, te in gkf.split(X, y, groups=mats):
        clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=4)
        clf.fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    print(f"\n按材质分组的 5 折（训练/测试不共享材质）：")
    print(f"  包识别准确率 {np.mean(accs):.3f}  各折 {[round(a,3) for a in accs]}")
    print("\n判读：显著高于多数类基准 -> 风格确实存在于单张瓦片，")
    print("      「包级一致性」是真问题；接近基准 -> 伪命题，这条路也停。")


if __name__ == "__main__":
    main()
