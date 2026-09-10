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


def anisotropy(img: np.ndarray) -> float:
    """横纵梯度幅度之差，按均值归一（不受整体对比度影响）。

    周期检测走的是行/列廓线，因此它只对**有方向性**的结构有效。
    实测 4951 张真人瓦片：能检出周期的一组中位 0.287，检不出的一组 0.063；
    手挑方向性类别 1.090 vs 各向同性类别 0.032（`analysis/paired/aniso_gate.py`）。
    阈值 0.20 落在两组之间。
    （原注释写的 0.774/0.036 出处标的是 crop_failure.py，但那里算的是
    grad_v/grad_h 比值而非本函数的归一化量，现有代码复现不出，已更正。）
    """
    g = img @ W if img.ndim == 3 else img
    h = float(np.abs(np.diff(g, axis=1)).mean())
    v = float(np.abs(np.diff(g, axis=0)).mean())
    return abs(h - v) / max((h + v) / 2, 1e-9)


def dominant_period(img: np.ndarray, lo: int = 8,
                    hi_frac: float = 0.625) -> float:
    """估计图中结构的主周期（源图像素）。行/列廓线自相关取首个显著峰。

    砖墙、木板这类材质的结构是周期性的，周期就是"一块砖多宽"。

    `hi_frac` 从 0.5 放宽到 **0.625**：0.5 时最大可搜周期正好是边长的一半，
    把"只有 2 层"的材质排除在外——`default_stone_brick` 因此只检出 6/40。
    放到 0.625 后变成 36/40，整体检出率 49%→63%
    （`analysis/paired/crop_failure.py`）。再放宽只涨误检不涨命中。
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


UNITS_PER_TILE = 4.5   # 真人在 16–32 上保持的结构单元数，见下


def _small(win: np.ndarray, size: int) -> np.ndarray:
    """窗口 -> size x size（量化前），与出厂管线同一条降采样路径。"""
    from PIL import Image
    return np.asarray(Image.fromarray(win.astype(np.uint8))
                      .resize((size,) * 2, Image.BOX)).astype(float)


def seam_stats(t: np.ndarray) -> tuple[float, float]:
    """(seam, internal)：平铺时露出来的那条缝，与内部相邻差的中位。

    贴图是**平铺**使用的，玩家看到的是一片墙而不是一张瓦片。
    首列与末列（首行与末行）接不上，平铺后就是一条贯穿整面墙的断线。
    """
    t = t.astype(float)
    seam = float(np.abs(t[:, -1] - t[:, 0]).mean() + np.abs(t[-1, :] - t[0, :]).mean())
    dv = [float(np.abs(t[:, i + 1] - t[:, i]).mean()) for i in range(t.shape[1] - 1)]
    dh = [float(np.abs(t[i + 1, :] - t[i, :]).mean()) for i in range(t.shape[0] - 1)]
    return seam, float(np.median(dv + dh))


def seam_offset(img: np.ndarray, side: int, size: int = 16,
                step: int = 12, keep: float = 0.85) -> tuple[int, int]:
    """给定窗口边长，选**位置**：在保住结构的窗口里取平铺接缝最小的。

    `auto_crop` 的 `side` 一直选得很讲究，**位置却写死居中**。
    位置对平铺使用直接可见：窗口没落在周期格点上，墙面就会有断线。

    ⚠ **`keep` 这条约束不是装饰**：纯平窗口的 seam/internal 是 **0**
    （分子分母都为零），即平坦窗口拿满分——单看接缝必然把画面拍平，
    正是本项目栽过三次的坑。故要求候选窗口的 internal >= 居中窗口的 `keep` 倍。
    居中窗口自己恒满足，候选集非空，最差退回居中。

    实证（`analysis/paired/seam_crop.py`，判据预注册 `95e5c81`）：
      操作检验 接缝比中位 **2.76 -> 0.85**，22/22 全降，p=4.8e-7；
              平坦守卫 internal 10.37 -> 10.72（**103%**，反而更有结构）
      ⚠ 那个 2.76 是**随机可行位置**的中位，不是居中的 3.34——见下面的零假设修正。
      主判据   3x3 平铺视图下经验证判官（opus-5）选接缝裁 **10/12 = 83%**
              （p=0.0386，[56%,96%]）
      ⚠ **但人工盲比没能确认它**（`study_seam.csv`，判据 `d4aad16`）：
              57 对里 **34/55 = 62%，p=0.105**，按预注册**主判据不成立**。
              且该次标注**位置偏好显著**（选左 71%，p=0.0027；接缝在左时被选 82%、
              在右时 41%），用时中位 2.9s——**仪器噪声大**，
              缺陷是我没给人配 VLM 那套"正反两问、不一致弃用"的去偏协议。
              **所以「更好看」不成立**：站得住的只有"接得上"这个可测性质。
      次判据   单张瓦片上 15 判 13 不一致 —— **收益只在平铺视图里显形**，
              这正是"赢的是接缝而不是别的画质差异"的旁证。

    ⚠ **零假设修正**（`analysis/paired/seam_null.py`，预注册 `863ba3f`）：
      上面两个 p 原先比的是「约 3930 个可行位置里的 argmin」对「居中这一个固定点」，
      而符号检验的 p=0.5 零假设对这种比法**不成立**——几千个里挑最小的，
      本就该赢过其中任一个。正确的零假设是**随机抽的可行位置**
      （代理若不传过量化，被选中的位置在出厂瓦片上就该与随机位置无异）。
      重跑（源图按同种子重渲染，逐材质逐位复现 22/22、35/35）：
        选中 vs 随机可行位置中位：**22/22 与 35/35**，p 与上面两个**逐位相同**；
        随机可行位置 vs 居中：15/22 p=0.134、22/35 p=0.175 —— **不显著**，
        即"居中"并不是个特别差的位置（它在随机位置里约排 60 分位）。
      **结论不变，但效应量要按 2.76->0.85 / 2.99->1.04 报**，
      不是 3.34->0.85 / 3.54->1.04：那段差是居中与典型位置之差，且不显著。

    ⚠ **泛化只到测量层面**（`seam_crop60.json`，60 个零重叠材质，预注册 `30b1fcf`）：
      接缝比 **2.99 -> 1.04**、**35/35** 全降、p=5.8e-11，守卫 117%——操作比首轮还硬；
      但 3x3 平铺下判官只有 **12/18 = 67%，p=0.238**，**没复现**。
      两轮 57 个材质接缝无一例外变好，判官不一致率却都在三成以上（33%/38%，
      单张 87%）——**是判官分辨力到头，不是效应消失**，但按协议不能宣称泛化。
      故：**"接得上"是客观且普遍的，"判官更喜欢"只在原 42 条上成立。**
      任何注释或正文都不许写成"经判官验证在任意材质上更好"。
    """
    H, W = img.shape[:2]
    ys = list(range(0, H - side + 1, step)) or [0]
    xs = list(range(0, W - side + 1, step)) or [0]
    cy, cx = (H - side) // 2, (W - side) // 2
    floor = keep * seam_stats(_small(img[cy:cy + side, cx:cx + side], size))[1]
    best, arg = None, (cy, cx)
    for y in ys:
        for x in xs:
            s_, i_ = seam_stats(_small(img[y:y + side, x:x + side], size))
            if i_ < floor:                 # 结构比现行裁法还少，不要
                continue
            r = s_ / max(i_, 1e-6)
            if best is None or r < best:
                best, arg = r, (y, x)
    return arg


def auto_crop(img: np.ndarray, size: int = 16, target_px: float | None = None,
              min_frac: float = 0.08,
              min_aniso: float = 0.20,
              seam_align: bool = False) -> tuple[np.ndarray, float]:
    """按结构尺度裁剪，使一个结构周期约占 `target_px` 个输出像素。

    B4 定位的失败原因：SDXL 在 1024 上画了约 25 层砖，
    要塞进 16 个输出像素——每像素 1.5 层，**超出奈奎斯特极限**，
    任何降采样器都表示不了（`experiments/downsamplers.png` 五种全败）。
    而 Minecraft 的 16x16 砖材质只有约 4 层。

    所以裁一块边长 = 周期 × size / target_px 的区域再降采样。
    实测 1024 的渲染图裁到 1/4–1/6 时结构恢复（`experiments/crop_scale.png`）。

    `target_px` 缺省按 `size / UNITS_PER_TILE` 算，即**每张图约 4.5 个结构单元**。
    这个数是从真人瓦片量出来的，不是猜的
    （`analysis/paired/resolution_tiers.py`，4951+679+349 张）：

    | 真人瓦片尺寸 | 主周期中位 | 每张图单元数 |
    | --- | --- | --- |
    | 16 | 5.0 px | 3.20 |
    | 32 | 10.0 px | 3.20 |
    | 64 | 8.5 px | 7.53 |

    **16 与 32 上单元数恒定**（真人把单元画大而不是加密），到 64 才加密。
    我们的目标区间 16–32 正好落在恒定段。

    **（2026-09-07 更正）** 上表原先是 4.0 / 4.6 / 9.1，那是 `hi_frac=0.5`
    量的——即本模块弃用前的旧设置。`UNITS_PER_TILE = 4.5` 就是从那组数定的，
    而 `auto_crop` 调 `dominant_period` 走的是默认 **0.625**：
    **定常数的口径与用常数的口径不一致**。生产口径下真人是 3.20，
    B9 的 125 个高分源独立给出 3.33，两者一致。
    常数**没有**跟着重调，全部已报结果都出自 4.5 的流水线；
    改成 3.2 会不会更好未测。改之前必须重跑盲比，别直接改这个数。

    返回 (裁剪后的图, 实际裁剪比例)。周期估不出来时原样返回。
    """
    if target_px is None:
        target_px = size / UNITS_PER_TILE
    H, Wd = img.shape[:2]
    # **双条件门**：检出周期 **且** 结构有方向性。
    # 单靠周期在放宽 hi_frac 后误检率会涨（沙、砾石、树冠会被误判有周期），
    # 而颗粒材质裁剪是有害的（B5）。加上各向异性后两个方向同时改善。
    # 数字以 `analysis/paired/aniso_gate.py` 的可复跑口径为准（论文 §5.3 同源）：
    #   仅周期 hi_frac=0.5（旧）      命中 80%  误检 14%
    #   仅周期 hi_frac=0.625          命中 98%  误检 20%
    #   双条件（0.625 + aniso≥0.20）  命中 96%  误检  4%
    # 留出集（按材质名词法规则划分、与调阈值那 6+8 类不重叠，31+101 类/1234 张）：
    # 仅周期 71%/39% → 双条件 83%/18%——泛化住了。
    # （旧注释的 70→90/6、留出 93/16 出自已遗失的手挑清单，不可复跑，已弃用。）
    per = dominant_period(img)
    if per > 0 and anisotropy(img) < min_aniso:
        per = 0.0
    if per <= 0:
        # **颗粒材质不裁。** 散布矿脉这类没有周期，
        # 按特征尺度折算（4 像素或 2 像素两种都试过）在图上都不如不裁——
        # 裁到最后只剩一颗，反而丢掉"散布"这个特征
        # （`experiments/autocrop.png`、`autocrop2.png`）。
        return img, 1.0
    side = per * size / max(target_px, 1e-6)
    # **上钳位会伪装成「门未触发」**：周期大到 4.5 个周期超过源图边长时，
    # side 被钳成整图，frac=1.0，与门主动拒绝的返回值一模一样。
    # 这是**源图尺寸限制**（补救是渲染更大或少放几个周期），不是适用性判断。
    # 实测 1024 下未触发的 69 例里有 21 例属于此类（论文 §5.3 已分开报）。
    # 调用方要区分的话，看 period*UNITS_PER_TILE 是否 >= min(H, W)。
    side = int(np.clip(side, min_frac * min(H, Wd), min(H, Wd)))
    if seam_align:
        # **只换位置，不换 side**：门与尺度完全不变，故 B9/B13/B14 全部沿用。
        # 默认关着是为了**保住已发表结果的可复现性**——所有已报数字都出自居中裁；
        # 交付脚本显式打开（`paint_region.py`、`batch_pack.py`）。
        y0, x0 = seam_offset(img, side, size)
    else:
        y0, x0 = (H - side) // 2, (Wd - side) // 2
    return img[y0:y0 + side, x0:x0 + side], side / min(H, Wd)
