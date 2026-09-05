"""降采样器：BOX 取平均会抹掉砖缝这类局部极值。

B4 定位到失败原因是尺度——SDXL 在 1024 上画的一块砖约 200 像素，
降到 16×16 后砖缝不足 1 像素，**BOX 平均直接抹平**。
砖缝在图里是局部极值，而平均正是消灭极值的操作。

这里实现几种保极值/保结构的降采样，接口统一：(高分图, size) -> (size,size,3)。
参考方向：Kopf 2013 内容自适应降采样、Öztireli 2015 感知降采样。
"""

import numpy as np

W = np.array([0.299, 0.587, 0.114])


def _cells(img: np.ndarray, size: int):
    """把图切成 size x size 个格子，返回 (size,size,h,w,3) 视图列表。"""
    H, Wd = img.shape[:2]
    ys = np.linspace(0, H, size + 1).round().astype(int)
    xs = np.linspace(0, Wd, size + 1).round().astype(int)
    for i in range(size):
        for j in range(size):
            yield i, j, img[ys[i]:ys[i+1], xs[j]:xs[j+1]]


def box(img, size):
    out = np.zeros((size, size, 3))
    for i, j, c in _cells(img, size):
        out[i, j] = c.reshape(-1, 3).mean(0)
    return out


def median(img, size):
    out = np.zeros((size, size, 3))
    for i, j, c in _cells(img, size):
        out[i, j] = np.median(c.reshape(-1, 3), 0)
    return out


def extremum(img, size, k=0.35):
    """每格取「偏离全局中位最远」的那一档分位，保住细的暗线/亮线。

    k 是分位位置：格内亮度分布上，若整体偏暗就取低分位、偏亮取高分位。
    这样细砖缝（格内少数极暗像素）不会被多数亮像素平均掉。
    """
    g = img.reshape(-1, 3) @ W
    lo, hi = np.percentile(g, 10), np.percentile(g, 90)
    out = np.zeros((size, size, 3))
    for i, j, c in _cells(img, size):
        f = c.reshape(-1, 3)
        lum = f @ W
        # 格内亮度相对全图的位置决定往哪边偏
        t = (lum.mean() - lo) / max(hi - lo, 1e-6)
        q = k if t > 0.5 else 1 - k
        out[i, j] = f[np.argsort(lum)[min(int(q * (len(f) - 1)), len(f) - 1)]]
    return out


def contrast_weighted(img, size, gamma=3.0):
    """按「偏离格均值的程度」加权平均：偏离越大权重越高。

    介于 BOX 与 extremum 之间——保住极值又不像 extremum 那样丢掉整体色调。
    gamma 控制强度，1 退化为接近 BOX。
    """
    out = np.zeros((size, size, 3))
    for i, j, c in _cells(img, size):
        f = c.reshape(-1, 3).astype(float)
        lum = f @ W
        d = np.abs(lum - lum.mean())
        w = (d / max(d.max(), 1e-6)) ** gamma + 1e-3
        out[i, j] = (f * w[:, None]).sum(0) / w.sum()
    return out


def bimodal(img, size):
    """格内亮度做二分（大津阈值），取占比大的那一簇的均值；
    若两簇亮度差很大而暗簇占比不低，则取暗簇——砖缝就是这种情况。"""
    out = np.zeros((size, size, 3))
    for i, j, c in _cells(img, size):
        f = c.reshape(-1, 3).astype(float)
        lum = f @ W
        if len(f) < 4 or lum.std() < 1e-6:
            out[i, j] = f.mean(0)
            continue
        th = lum.mean()
        for _ in range(8):                    # 简易 k=2 一维 Lloyd
            a, b = lum[lum <= th], lum[lum > th]
            if not len(a) or not len(b):
                break
            th = (a.mean() + b.mean()) / 2
        dark, bright = lum <= th, lum > th
        if dark.sum() and bright.sum():
            gap = lum[bright].mean() - lum[dark].mean()
            frac = dark.sum() / len(lum)
            take = dark if (gap > 25 and frac > 0.25) else (
                dark if frac > 0.5 else bright)
        else:
            take = np.ones(len(lum), bool)
        out[i, j] = f[take].mean(0)
    return out


METHODS = {"box": box, "median": median, "extremum": extremum,
           "contrast": contrast_weighted, "bimodal": bimodal}


def dominant_period(img: np.ndarray, lo: int = 8, hi_frac: float = 0.5) -> float:
    """估计图中结构的主周期（源图像素）。行/列廓线自相关取首个显著峰。

    砖墙、木板这类材质的结构是周期性的，周期就是"一块砖多宽"。
    """
    g = img @ W if img.ndim == 3 else img
    best = []
    for prof in (g.mean(1), g.mean(0)):
        p = prof - prof.mean()
        if p.std() < 1e-6:
            continue
        ac = np.correlate(p, p, "full")[len(p) - 1:]
        ac = ac / (ac[0] + 1e-9)
        hi = int(len(p) * hi_frac)
        if hi <= lo + 2:
            continue
        seg = ac[lo:hi]
        # 首个局部极大且高于 0.2 的滞后
        for i in range(1, len(seg) - 1):
            if seg[i] > seg[i-1] and seg[i] >= seg[i+1] and seg[i] > 0.2:
                best.append(lo + i)
                break
    return float(np.median(best)) if best else 0.0


def correlation_length(img: np.ndarray, thresh: float = 0.5) -> float:
    """自相关首次跌破 `thresh` 的滞后——特征尺度，对非周期材质也成立。

    `dominant_period` 只对周期结构有效（砖、木板）；
    矿石这类散布颗粒没有周期，但有典型斑块大小，衰减长度量的就是它。
    """
    g = img @ W if img.ndim == 3 else img
    ls = []
    for prof in (g.mean(1), g.mean(0)):
        p = prof - prof.mean()
        if p.std() < 1e-6:
            continue
        ac = np.correlate(p, p, "full")[len(p) - 1:]
        ac = ac / (ac[0] + 1e-9)
        below = np.nonzero(ac < thresh)[0]
        if len(below):
            ls.append(float(below[0]))
    return float(np.median(ls)) if ls else 0.0


def auto_crop(img: np.ndarray, size: int = 16, target_px: float = 4.0,
              min_frac: float = 0.08) -> tuple[np.ndarray, float]:
    """按结构尺度裁剪，使一个结构周期约占 `target_px` 个输出像素。

    B4 定位的失败原因：SDXL 在 1024 上画了约 25 层砖，
    要塞进 16 个输出像素——每像素 1.5 层，**超出奈奎斯特极限**，
    任何降采样器都表示不了（`experiments/downsamplers.png` 五种全败）。
    而 Minecraft 的 16x16 砖材质只有约 4 层。

    所以裁一块边长 = 周期 × size / target_px 的区域再降采样。
    实测 1024 的渲染图裁到 1/4–1/6 时结构恢复（`experiments/crop_scale.png`）。

    返回 (裁剪后的图, 实际裁剪比例)。周期估不出来时原样返回。
    """
    H, Wd = img.shape[:2]
    per = dominant_period(img)
    if per <= 0:
        # **颗粒材质不裁。** 散布矿脉这类没有周期，
        # 按特征尺度折算（4 像素或 2 像素两种都试过）在图上都不如不裁——
        # 裁到最后只剩一颗，反而丢掉"散布"这个特征
        # （`experiments/autocrop.png`、`autocrop2.png`）。
        return img, 1.0
    side = per * size / max(target_px, 1e-6)
    side = int(np.clip(side, min_frac * min(H, Wd), min(H, Wd)))
    y0, x0 = (H - side) // 2, (Wd - side) // 2
    return img[y0:y0 + side, x0:x0 + side], side / min(H, Wd)
