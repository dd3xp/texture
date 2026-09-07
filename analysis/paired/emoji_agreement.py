"""B1 是否推广到像素画之外：同一 emoji 的多套独立重绘。

B1/B7 在材质包上量到：同一材质的独立作者逐格一致率 0.098，
仅比配对零假设 0.086 高 1.2 个百分点——逐像素几乎无信号。

若这只在像素画成立，它是个特例；若在**同一内容的多套独立重绘**上同样成立，
它就是"参考不唯一时逐像素评测失效"的一般结论。

三套 emoji 各自独立设计、许可允许研究使用：
Noto（Apache/OFL）、Twemoji（CC-BY 4.0）、OpenMoji（CC BY-SA 4.0）。

口径与 B1 **完全一致**：降到 16×16、按亮度量化到 12 档、
配对零假设 = agree(A, shuffle(B))。
"""

import argparse
import io
import itertools
from pathlib import Path

import numpy as np
import requests
from PIL import Image

SETS = {
    "noto": "https://raw.githubusercontent.com/googlefonts/noto-emoji/main/png/32/emoji_u{lo}.png",
    "twemoji": "https://raw.githubusercontent.com/jdecked/twemoji/main/assets/72x72/{lo}.png",
    "openmoji": "https://raw.githubusercontent.com/hfg-gmuend/openmoji/master/color/72x72/{up}.png",
}
# 常见单码位 emoji，避开组合序列（各套命名不一致）
CODES = ["1f600","1f603","1f604","1f606","1f609","1f60a","1f60d","1f618",
         "1f61c","1f620","1f622","1f62d","1f631","1f634","1f637","1f644",
         "1f383","1f384","1f385","1f386","1f388","1f389","1f38a","1f38d",
         "1f34e","1f34f","1f350","1f351","1f352","1f353","1f345","1f346",
         "1f347","1f348","1f349","1f34a","1f34b","1f34c","1f34d","1f336",
         "1f680","1f681","1f682","1f683","1f684","1f685","1f686","1f687",
         "1f30d","1f30e","1f30f","1f311","1f312","1f313","1f314","1f315",
         "1f408","1f415","1f416","1f417","1f418","1f419","1f41a","1f41b"]
W = np.array([0.299, 0.587, 0.114])


def fetch(cache: Path, name: str, code: str):
    f = cache / f"{name}_{code}.png"
    if f.exists():
        return f if f.stat().st_size > 0 else None
    url = SETS[name].format(lo=code, up=code.upper())
    try:
        r = requests.get(url, timeout=25)
    except Exception:
        return None
    f.write_bytes(r.content if r.status_code == 200 else b"")
    return f if r.status_code == 200 else None


def quantize(f: Path, n: int = 16, k: int = 12):
    """降到 n×n，按亮度量化到 k 档——与 B1 口径一致。
    透明像素按白底合成（各套背景处理不同，不合成会引入伪差异）。

    同时返回**前景掩码**（合成后仍有 alpha 的格子）。这是必须的：
    emoji 画布四周大片透明，合成到白底后天然一致，
    不排除它们会把差值虚高约一个数量级（B16）。"""
    im = Image.open(f).convert("RGBA").resize((n, n), Image.BOX)
    a = np.asarray(im).astype(float)
    alpha = a[..., 3] / 255.0
    rgb = a[..., :3] * alpha[..., None] + 255.0 * (1 - alpha[..., None])
    lum = rgb @ W
    q = np.clip((lum / 255.0 * (k - 1)).round(), 0, k - 1)
    return q, alpha > 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/emoji"))
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)

    have = {}
    for c in CODES:
        got = {n: fetch(args.cache, n, c) for n in SETS}
        if all(v is not None for v in got.values()):
            have[c] = got
    print(f"三套齐全的 emoji：{len(have)}/{len(CODES)}")
    if len(have) < 10:
        print("样本太少，停"); return

    rng = np.random.default_rng(0)
    real, null = [], []
    freal, fnull, bgfrac = [], [], []
    for c, files in have.items():
        arrs = {n: quantize(f) for n, f in files.items()}
        for a, b in itertools.combinations(arrs, 2):
            (x, mx), (y, my) = arrs[a], arrs[b]
            real.append(float((x == y).mean()))
            null.append(float((x == rng.permutation(y.ravel()).reshape(y.shape)).mean()))
            # 前景口径：取两套前景的交集，零假设在**掩码内**洗牌
            m = mx & my
            bgfrac.append(1.0 - float(m.mean()))
            if m.sum() < 8:
                continue
            xv, yv = x[m], y[m]
            freal.append(float((xv == yv).mean()))
            fnull.append(float((xv == rng.permutation(yv)).mean()))
    real, null = np.array(real), np.array(null)
    freal, fnull = np.array(freal), np.array(fnull)
    from scipy import stats

    def report(tag, r, n):
        d = float(np.median(r) - np.median(n))
        print(f"  {tag:<10} 真实 {np.median(r):.4f}   零假设 {np.median(n):.4f}   差 {d:+.4f}   低于零假设 {(r<n).mean():.1%}   Wilcoxon p={stats.wilcoxon(r,n).pvalue:.3g}")
        return d

    print(f"\n套间两两配对 {len(real)} 组（3 套 -> 每个 emoji 3 对）")
    d_all = report("全格子", real, null)
    d_fg = report("仅前景", freal, fnull)
    print(f"  背景（两套前景交集之外）格子占比中位 {np.median(bgfrac):.1%}，前景口径有效组 {len(freal)}")
    print(f"\n对照 · 材质包（B1/B7）：真实 0.0977，零假设 0.0859，差 +0.0117")

    # 判读跟着数走。全格子口径把天然一致的透明背景算了进去，差值虚高一个数量级；
    # 能与材质包比较的是前景口径。此前这里无条件打印「也仅略高于」，
    # 而全格子差 +0.148 其实是材质包 +0.0117 的十几倍——与自己的输出矛盾。
    print()
    print(f"判读：全格子差 {d_all:+.4f} 是材质包 +0.0117 的 {d_all/0.0117:.0f} 倍——这个口径不可比，背景天然一致。")
    if abs(d_fg - 0.0117) < 0.01:
        print(f"      前景口径差 {d_fg:+.4f}，与材质包 +0.0117 同量级 -> 「参考不唯一时逐格几乎无信号」可推广。")
    else:
        print(f"      前景口径差 {d_fg:+.4f}，与材质包 +0.0117 不同量级 -> 不能声称推广，需说明差异来源。")


if __name__ == "__main__":
    main()
