"""材质能否从单张瓦片认出？——benchmark 第二个指标是否成立。

包级 benchmark 要同时测两件事，否则有平凡解：
  从目标包随便抄一张 -> 风格满分、材质全错
  从别的包抄同一材质 -> 材质满分、风格全错
风格那一头已验（`pack_identity.py`：42 类 0.444 vs 基准 0.085）。
这里验材质那一头。

**按包分组**做交叉验证：训练与测试不共享材质包，
逼分类器学"这是什么材质"而不是"这是谁画的"。
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np


def feats(idx: np.ndarray, pal: np.ndarray) -> np.ndarray:
    """材质靠空间排布，所以这里用**归一化的空间特征**，
    并刻意剥掉风格线索（绝对色值、色数）：把索引归一到 [0,1] 再取空间统计。"""
    a = idx.astype(float) / max(len(pal) - 1, 1)
    rows = a.mean(1)
    cols = a.mean(0)
    def acf(v):
        v = v - v.mean()
        if v.std() < 1e-9:
            return np.zeros(7)
        c = np.correlate(v, v, "full")[len(v) - 1:]
        return (c / (c[0] + 1e-9))[1:8]
    q = np.quantile(a, [0.1, 0.25, 0.5, 0.75, 0.9])
    return np.concatenate([
        rows - rows.mean(), cols - cols.mean(),          # 行/列廓线（去均值）
        acf(rows), acf(cols),                            # 行/列自相关
        q,                                               # 档位分布形状
        [float(np.abs(np.diff(a, axis=0)).mean()),
         float(np.abs(np.diff(a, axis=1)).mean()),
         float(a.std()),
         float((a[:, :-1] == a[:, 1:]).mean())],
    ])


def main():
    ds = json.loads(Path("data/tiles/dataset_k16.json").read_text())
    X, y, packs = [], [], []
    for s in ds["samples"]:
        if s["size"] != 16:
            continue
        pal = np.array(s["palette"], np.uint8)
        idx = np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16)
        X.append(feats(idx, pal)); y.append(s["material"]); packs.append(s["pack"])
    X, y, packs = np.array(X), np.array(y), np.array(packs)
    cnt = Counter(y)
    keep = {k for k, v in cnt.items() if v >= 6}
    sel = np.array([v in keep for v in y])
    X, y, packs = X[sel], y[sel], packs[sel]
    print(f"瓦片 {len(X)}，材质 {len(keep)}，包 {len(set(packs))}")
    print(f"多数类基准 {Counter(y).most_common(1)[0][1]/len(y):.4f}")
    print(f"随机基准 {1/len(keep):.4f}")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    accs, top5 = [], []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=packs):
        clf = RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=4)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])
        cls = clf.classes_
        accs.append(float((cls[p.argmax(1)] == y[te]).mean()))
        idx5 = np.argsort(-p, axis=1)[:, :5]
        top5.append(float(np.mean([y[te][i] in cls[idx5[i]] for i in range(len(te))])))
    print(f"\n按包分组的 5 折（训练/测试不共享材质包）：")
    print(f"  材质识别 top-1 {np.mean(accs):.3f}   top-5 {np.mean(top5):.3f}")
    print("\n判读：显著高于随机 -> 材质可从单张瓦片识别，可作为第二个指标；")
    print("      接近随机 -> 这个指标撑不住，benchmark 只能靠风格那一头 + 人工。")


if __name__ == "__main__":
    main()
