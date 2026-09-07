"""无依赖的精确统计——避免脚本因环境缺 scipy 在末尾崩溃。

服务器上有两个环境：`jzs_train` 有可用的 diffusers 但**没有 scipy**，
`SD-piXL` 有 scipy 但 diffusers 版本冲突。
需要 GPU 生成的脚本只能用前者，于是末尾的 `from scipy import stats`
已经导致三次崩溃（数据都已落盘，但统计行丢失、看起来像失败）。

这里只实现实际用到的两项，纯标准库。
"""

from math import comb, lgamma, log, exp


def binom_test(k: int, n: int, p: float = 0.5) -> float:
    """双尾精确二项检验。把概率不高于观测值的所有结果加起来。"""
    if n <= 0:
        return float("nan")
    pm = [comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return min(1.0, sum(x for x in pm if x <= pm[k] * (1 + 1e-9)))


def _beta_ppf(q: float, a: float, b: float, iters: int = 200) -> float:
    """Beta 分位数，二分法。用于 Jeffreys 区间。"""
    def cdf(x):
        # 正则化不完全 Beta，连分数展开
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
        front = exp(a * log(x) + b * log(1 - x) - lbeta) / a
        f, c, d = 1.0, 1.0, 0.0
        for i in range(0, 200):
            m = i // 2
            if i == 0:
                num = 1.0
            elif i % 2 == 0:
                num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
            else:
                num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
            d = 1.0 + num * d
            d = 1e-30 if abs(d) < 1e-30 else d
            d = 1.0 / d
            c = 1.0 + num / c
            c = 1e-30 if abs(c) < 1e-30 else c
            f *= c * d
            if abs(1.0 - c * d) < 1e-10:
                break
        return front * (f - 1.0)

    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if cdf(mid) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def jeffreys(k: int, n: int, alpha: float = 0.05):
    """Jeffreys 区间（Beta(k+.5, n-k+.5) 的分位数）。"""
    if n <= 0:
        return float("nan"), float("nan")
    return (_beta_ppf(alpha / 2, k + .5, n - k + .5),
            _beta_ppf(1 - alpha / 2, k + .5, n - k + .5))
