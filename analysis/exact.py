"""无依赖的精确统计——避免脚本因环境缺 scipy 在末尾崩溃。

服务器上有两个环境：`jzs_train` 有可用的 diffusers 但**没有 scipy**，
`SD-piXL` 有 scipy 但 diffusers 版本冲突。
需要 GPU 生成的脚本只能用前者，于是末尾的 `from scipy import stats`
已经导致三次崩溃（数据都已落盘，但统计行丢失、看起来像失败）。

这里只实现实际用到的两项，纯标准库。
"""

from math import lgamma, log, log1p, exp


def _logsumexp(vs) -> float:
    """对数空间求和。逐项 exp 再相加会让每一项各自下溢成 0。"""
    vs = [v for v in vs if v != -float("inf")]
    if not vs:
        return -float("inf")
    m = max(vs)
    return m + log(sum(exp(v - m) for v in vs))


def binom_test_log(k: int, n: int, p: float = 0.5) -> float:
    """双尾精确二项检验，返回**自然对数**的 p。

    真值小到 1e-308 以下时 `binom_test` 只能返回 0.0，而把 0.0 印成 "p=0"
    是一句不该发表的话。需要报极小 p 的地方用这个，配 `fmt_p` 打印。
    """
    if n <= 0:
        return float("nan")
    lg = lgamma(n + 1)

    def lpmf(i):
        if p <= 0.0:
            return 0.0 if i == 0 else -float("inf")
        if p >= 1.0:
            return 0.0 if i == n else -float("inf")
        return (lg - lgamma(i + 1) - lgamma(n - i + 1)
                + i * log(p) + (n - i) * log1p(-p))

    lv = [lpmf(i) for i in range(n + 1)]
    cut = lv[k] + log1p(1e-9)
    return min(0.0, _logsumexp([v for v in lv if v <= cut]))


def binom_test(k: int, n: int, p: float = 0.5) -> float:
    """双尾精确二项检验。把概率不高于观测值的所有结果加起来。

    pmf 走**对数空间**：原先写的是 `comb(n, i) * p ** i * ...`，`comb` 在
    n 上千时是几百位的大整数，乘上早已下溢成 0.0 的 `p ** i` 会直接抛
    `OverflowError: int too large to convert to float`。真人瓦片那种几千个
    样本的符号检验因此崩在脚本末尾——正是本模块开头说要避免的那种崩法
    （数据都算完了，只丢统计行）。对数空间下 n 多大都不溢出，
    小 n 的结果与原式在 1e-12 内一致。

    ⚠ **求和也必须在对数空间做**（2026-09-10 补）：此前是把每项 `exp` 回
    线性再相加，n=2971、k=241 时每一项各自下溢成 0，返回**恰好 0.0**。
    真值低于 1e-308 时本函数仍只能返回 0.0——**那不是零，是下限**。
    要报这种量级的 p，用 `binom_test_log`，打印用 `fmt_p`。
    """
    lp = binom_test_log(k, n, p)
    if lp != lp:                       # nan
        return lp
    return min(1.0, exp(lp) if lp > -745.0 else 0.0)


def fmt_p(k: int, n: int, p: float = 0.5, sig: int = 3) -> str:
    """把 p 打印成人能信的样子——低于双精度下限时写成**界**，不写成 0。"""
    lp = binom_test_log(k, n, p)
    if lp != lp:
        return "nan"
    if lp > -745.0:
        v = min(1.0, exp(lp))
        if v > 0.0:
            return f"{v:.{sig}g}"
    log10p = lp / 2.302585092994046
    return f"<1e{int(log10p) - 1}"


def _betainc_cf(x: float, a: float, b: float) -> float:
    """正则化不完全 Beta I_x(a,b) 的连分数展开（只在 x < (a+1)/(a+b+2) 时调用）。"""
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


def _beta_ppf(q: float, a: float, b: float, iters: int = 200) -> float:
    """Beta 分位数，二分法。用于 Jeffreys 区间。"""
    def cdf(x):
        # 正则化不完全 Beta。连分数只在 x < (a+1)/(a+b+2) 时收敛，其余用对称式
        # I_x(a,b) = 1 - I_{1-x}(b,a)（2026-09-11 修：缺这一步时 39/93 的 Jeffreys 上限算成了 1.0）
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        if x > (a + 1) / (a + b + 2):
            return 1.0 - _betainc_cf(1 - x, b, a)
        return _betainc_cf(x, a, b)

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
