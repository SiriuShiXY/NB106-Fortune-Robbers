import numpy as np
import math
import multiprocessing as mp
from deap import gp
from tqdm import tqdm

# Set multiprocessing start method
mp.set_start_method('spawn', force=True)
sign = np.sign   # set global sign to numpy sign function

LAMBDA1 = 0.4
LAMBDA2 = 2.0
ARITY = 5                # <-- if you want to change the window width, just change this + the registration place
EPS   = 1e-10

def compute_rank_ic(predictions: np.ndarray, target: np.ndarray) -> float:
    """Vectorised Pearson rank correlation (Spearman IC)."""
    pred_ranks = predictions.argsort().argsort()
    target_ranks = target.argsort().argsort()
    pred_centered = pred_ranks - pred_ranks.mean()
    target_centered = target_ranks - target_ranks.mean()
    numerator = np.dot(pred_centered, target_centered)
    denominator = np.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum())
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _group_returns(predictions: np.ndarray, target: np.ndarray, n_groups: int = 5) -> np.ndarray:
    """Split samples into *n_groups* by descending prediction values and return mean target in each group."""
    order = predictions.argsort()[::-1]  # descending sort
    group_sizes = np.full(n_groups, len(predictions) // n_groups)
    group_sizes[: len(predictions) % n_groups] += 1  # distribute remainder
    splits = np.cumsum(group_sizes)
    groups = np.split(order, splits[:-1])
    return np.array([target[idx].mean() if len(idx) > 0 else 0.0 for idx in groups])


def compute_topR(predictions: np.ndarray, target: np.ndarray, n_groups: int = 5) -> float:
    """TopR metric – see Eq.(6) in Scientific Reports (2024)."""
    group_ret = _group_returns(predictions, target, n_groups)
    topR = group_ret[0]
    flopR = group_ret[-1]
    mean_total = target.mean()
    return max(topR - mean_total, flopR - mean_total)


def compute_monotonicity(predictions: np.ndarray, target: np.ndarray, n_groups: int = 5) -> float:
    """Monotonicity metric – Eq.(7). Higher is better (max=1)."""
    group_ret = _group_returns(predictions, target, n_groups)
    diff_sign = np.sign(group_ret[:-1] - group_ret[1:])
    inc_monotone = np.clip(diff_sign, 0, None).mean()   # strictly decreasing group returns (good if higher scores ⇒ higher returns)
    dec_monotone = np.clip(-diff_sign, 0, None).mean()  # strictly increasing group returns (good if higher scores ⇒ lower returns)
    return max(inc_monotone, dec_monotone)

# Shared data containers for multiprocessing
global_x = None
global_y = None
global_toolbox = None

def init_worker(x, y, toolbox):
    """Initializer for multiprocessing pool. Stores shared arrays and toolbox in globals."""
    global global_x, global_y, global_toolbox
    global_x = x
    global_y = y
    global_toolbox = toolbox


def eval_fitness(individual, X=None, y=None, toolbox=None, n_groups: int = 5):
    """Evaluate an individual according to Eq.(8):
        Fitness = TopR + λ1·Monotonicity + λ2·RankIC
    Returns a tuple because DEAP expects that.
    """
    global global_x, global_y, global_toolbox

    if X is None:
        X = global_x
    if y is None:
        y = global_y
    if toolbox is None:
        toolbox = global_toolbox

    func = toolbox.compile(expr=individual)

    try:
        preds = np.array([func(*row) for row in X], dtype=np.float64)
        preds = np.nan_to_num(preds, nan=0.0, posinf=1e10, neginf=-1e10)
        preds = np.clip(preds, -1e10, 1e10)

        top_r = compute_topR(preds, y, n_groups)
        mono = compute_monotonicity(preds, y, n_groups)
        ric = compute_rank_ic(preds, y)

        fitness_value = top_r + LAMBDA1 * mono + LAMBDA2 * ric
    except Exception as err:
        print(f"[Fitness Error] {err}")
        fitness_value = -np.inf

    return (fitness_value,)


def parallel_evaluate(individuals, x, y, toolbox, n_jobs: int | None = None):
    """Vectorised parallel evaluation of *individuals* across *n_jobs* cores."""
    if n_jobs is None:
        n_jobs = mp.cpu_count()

    chunk_size = max(1, len(individuals) // (n_jobs * 4))

    with mp.Pool(processes=n_jobs, initializer=init_worker, initargs=(x, y, toolbox)) as pool:
        fitnesses = list(
            tqdm(
                pool.imap(eval_fitness, individuals, chunksize=chunk_size),
                total=len(individuals),
                desc="Evaluating individuals",
                unit="ind",
            )
        )
    return fitnesses

# ---------- heuristic operators ----------
def protected_division(x1, x2):
    return 1.0 if abs(x2) < 1e-10 else x1 / x2

def protected_sqrt(x):
    return math.sqrt(abs(x))

def protected_log(x):
    return 0.0 if x == 0 else math.log(abs(x))

def safe_exp(x):
    if x > 700:
        return 1e308
    if x < -700:
        return 0.0
    return math.exp(x)

def rank(*xs):              # absolute ranking 
    arr = np.array(xs)
    return float(arr.argsort().argsort()[-1] + 1) / len(arr)

def decay_linear(*xs):
    w = np.arange(1, len(xs) + 1)
    return float(np.dot(w, xs) / w.sum())

def winsorize(*xs, p=0.05):
    q_lo, q_hi = np.quantile(xs, [p, 1 - p])
    clipped = np.clip(xs, q_lo, q_hi)
    return float(clipped[-1])          # retunr the last value after winsorization

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def ts_max(*xs):   return float(np.nanmax(xs))
def ts_min(*xs):   return float(np.nanmin(xs))
def ts_sum(*xs):   return float(np.nansum(xs))
def ts_mean(*xs):  return float(np.nanmean(xs))
def ts_prod(*xs):  return float(np.prod(xs))

def ts_nanmean(*xs): return float(np.nanmean(xs))
def ts_stddev(*xs):  return float(np.nanstd(xs, ddof=1))
def ts_skewness(*xs):
    a = np.array(xs); mu = np.nanmean(a); sigma = np.nanstd(a)
    if sigma < EPS: return 0.0
    return float(np.nanmean(((a - mu) / sigma) ** 3))
def ts_kurtosis(*xs):
    a = np.array(xs); mu = np.nanmean(a); sigma = np.nanstd(a)
    if sigma < EPS: return 0.0
    return float(np.nanmean(((a - mu) / sigma) ** 4) - 3)

def delta(x1, x2, *rest):   # x1 = t, x2 = t-1
    return x1 - x2

def ts_max_diff(*xs):
    return float(np.nanmax(xs) - np.nanmin(xs))

def ts_min_diff(*xs):
    a = np.array(xs)
    return float(np.nanmin(np.diff(a)))

def ts_return(x1, x2, *rest):
    return 0.0 if abs(x2) < EPS else (x1 / x2 - 1.0)

def rank_add(a, b): return rank(a + b)
def rank_sub(a, b): return rank(a - b)
def rank_mul(a, b): return rank(a * b)
def rank_div(a, b): return rank(protected_division(a, b))

def ts_corr(x1, x2, x3, x4, x5):
    a, b = np.array([x1, x2, x3, x4, x5]), np.array([x2, x3, x4, x5, x1])
    if np.nanstd(a) < EPS or np.nanstd(b) < EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def ts_cov(x1, x2, x3, x4, x5):
    a, b = np.array([x1, x2, x3, x4, x5]), np.array([x2, x3, x4, x5, x1])
    return float(np.nanmean((a - a.mean()) * (b - b.mean())))

def ts_argmax(*xs): return float(np.nanargmax(xs) + 1) / len(xs)
def ts_argmin(*xs): return float(np.nanargmin(xs) + 1) / len(xs)

def zscore(x, mu=0.0, sigma=1.0):
    return 0.0 if abs(sigma) < EPS else (x - mu) / sigma

def ts_zscore(*xs):
    a = np.array(xs); mu = np.nanmean(a); sigma = np.nanstd(a)
    if sigma < EPS: return 0.0
    return float((a[-1] - mu) / sigma)

def ts_scale(*xs):
    a = np.array(xs)
    lo, hi = np.nanmin(a), np.nanmax(a)
    return 0.0 if hi - lo < EPS else float((a[-1] - lo) / (hi - lo))

def ts_min_max_cps(*xs):
    """center-plus-scale: [-1,1] 中心化"""
    a = np.array(xs); lo, hi = np.nanmin(a), np.nanmax(a)
    if hi - lo < EPS: return 0.0
    return float(2 * (a[-1] - lo) / (hi - lo) - 1)

def ts_ir(*xs):
    a = np.array(xs); mu = np.nanmean(a); sigma = np.nanstd(a, ddof=1)
    return 0.0 if sigma < EPS else float(mu / sigma)

def delay(x1, x2, *rest):   # return t-1 value
    return x2

def ts_returns_mean(*xs):
    """calculate mean returns"""
    if len(xs) < 2:
        return 0.0
    a = np.array(xs)
    diffs = np.diff(a)
    denoms = a[1:]
    mask = np.abs(denoms) > EPS
    if not np.any(mask):
        return 0.0
    rets = np.zeros_like(diffs, dtype=float)
    rets[mask] = diffs[mask] / denoms[mask]
    return float(np.nanmean(rets))

def ts_returns_std(*xs):
    """calculate returns volatility"""
    if len(xs) < 2:
        return 0.0
    a = np.array(xs)
    diffs = np.diff(a)
    denoms = a[1:]
    mask = np.abs(denoms) > EPS
    if not np.any(mask):
        return 0.0
    rets = np.zeros_like(diffs, dtype=float)
    rets[mask] = diffs[mask] / denoms[mask]
    return float(np.nanstd(rets))

def ts_rank_mean(*xs):
    """Calculate normalized rank mean"""
    arr = np.array(xs)
    ranks = arr.argsort().argsort() / (len(arr) - 1)  # normalized ranks
    return float(np.nanmean(ranks))

def ts_rank_std(*xs):
    """Calculate normalized rank standard deviation"""
    arr = np.array(xs)
    ranks = arr.argsort().argsort() / (len(arr) - 1)
    return float(np.nanstd(ranks))

def ts_ma(*xs):
    """Calculate moving average"""
    return float(np.nanmean(xs))

def ts_ema(*xs, alpha=0.1):
    """Calculate exponential moving average"""
    weights = (1 - alpha) ** np.arange(len(xs))[::-1]
    weights /= weights.sum()
    return float(np.nansum(weights * xs))

def ts_wma(*xs):
    """Calculate weighted moving average"""
    weights = np.arange(1, len(xs) + 1)
    weights = weights / weights.sum()
    return float(np.nansum(weights * xs))

def ts_momentum(*xs):
    """Calculate momentum"""
    if len(xs) < 2:
        return 0.0
    return float((xs[0] / xs[-1]) - 1) if abs(xs[-1]) > EPS else 0.0

def ts_rsi(*xs):
    """Calculate Relative Strength Index (RSI)"""
    if len(xs) < 2:
        return 50.0
    diff = np.diff(xs)
    pos_gains = np.clip(diff, 0, None)
    neg_losses = np.clip(-diff, 0, None)
    avg_gain = np.nanmean(pos_gains)
    avg_loss = np.nanmean(neg_losses)
    if avg_loss < EPS:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))

def ts_beta(*xs):
    """Calculate beta coefficient"""
    if len(xs) < 2:
        return 0.0
    x = np.array(xs[:-1])  # current period
    y = np.array(xs[1:])   # lag period
    if np.nanstd(y) < EPS:
        return 0.0
    return float(np.nanmean((x - np.nanmean(x)) * (y - np.nanmean(y))) / np.nanvar(y))

def ts_autocorr(*xs):
    """Calculate autocorrelation"""
    if len(xs) < 2:
        return 0.0
    x = np.array(xs[:-1])
    y = np.array(xs[1:])
    if np.nanstd(x) < EPS or np.nanstd(y) < EPS:
        return 0.0
    return float(np.corrcoef(x, y)[0,1])

def ts_half_life(*xs):
    """Calculate half-life of mean reversion"""
    if len(xs) < 3:
        return 1.0
    y = np.diff(xs)
    x = xs[:-1]
    if np.nanstd(x) < EPS:
        return 1.0
    beta = np.nanmean(x * y) / np.nanvar(x)
    if beta >= 0:
        return 1.0
    return float(-np.log(2) / beta)

def ts_volatility(*xs):
    """Calculate volatility (standard deviation of returns)"""
    if len(xs) < 2:
        return 0.0
    a = np.array(xs)
    diffs = np.diff(a)
    denoms = a[1:]
    mask = np.abs(denoms) > EPS
    if not np.any(mask):
        return 0.0
    rets = np.zeros_like(diffs, dtype=float)
    rets[mask] = diffs[mask] / denoms[mask]
    return float(np.nanstd(rets))

def ts_downside_std(*xs):
    """Calculate downside standard deviation"""
    if len(xs) < 2:
        return 0.0
    a = np.array(xs)
    diffs = np.diff(a)
    denoms = a[1:]
    mask = np.abs(denoms) > EPS
    if not np.any(mask):
        return 0.0
    rets = np.zeros_like(diffs, dtype=float)
    rets[mask] = diffs[mask] / denoms[mask]
    downside = rets[rets < 0]
    return float(np.nanstd(downside)) if len(downside) > 0 else 0.0

def ts_skew(*xs):
    """Calculate skewness"""
    a = np.array(xs)
    if len(a) < 2:
        return 0.0
    mu = np.nanmean(a)
    sigma = np.nanstd(a)
    if sigma < EPS:
        return 0.0
    return float(np.nanmean(((a - mu) / sigma) ** 3))

def ts_kurt(*xs):
    """Calculate kurtosis"""
    a = np.array(xs)
    if len(a) < 2:
        return 0.0
    mu = np.nanmean(a)
    sigma = np.nanstd(a)
    if sigma < EPS:
        return 0.0
    return float(np.nanmean(((a - mu) / sigma) ** 4))


__all__ = [
    # 基础
    "protected_division", "protected_sqrt", "protected_log", "safe_exp",
    # Heuristic ops
    "rank", "decay_linear", "winsorize", "sigmoid",
    "ts_max", "ts_min", "ts_sum", "ts_mean", "ts_prod", "ts_nanmean",
    "ts_stddev", "ts_skewness", "ts_kurtosis",
    "delta", "ts_max_diff", "ts_min_diff", "ts_return",
    "rank_add", "rank_sub", "rank_mul", "rank_div",
    "ts_corr", "ts_cov", "ts_argmax", "ts_argmin",
    "ts_zscore", "zscore", "ts_scale", "ts_min_max_cps",
    "ts_ir", "delay", "sign",
    # Alpha Operators
    "ts_returns_mean", "ts_returns_std",
    "ts_rank_mean", "ts_rank_std",
    "ts_ma", "ts_ema", "ts_wma",
    "ts_momentum", "ts_rsi", "ts_beta",
    "ts_autocorr", "ts_half_life",
    "ts_volatility", "ts_downside_std",
    "ts_skew", "ts_kurt",
    # Constants
    "ARITY"
]
