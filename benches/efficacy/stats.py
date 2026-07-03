#!/usr/bin/env python3
"""Dependency-free statistics for the Reflex efficacy study (REF-176 Phase 3).

The whole efficacy harness is stdlib-only on purpose (numpy/scipy are not
installed in the benchmark environment and pinning them would make the analysis
harder to reproduce on a fresh checkout). So the non-parametric machinery the
plan calls for — **bootstrap confidence intervals** and the **Wilcoxon
signed-rank test** — is implemented here from scratch, deterministically.

Determinism is a hard requirement: the same metrics table must always produce
the same CIs and p-values. Every bootstrap uses an explicit, seeded
``random.Random`` instance; nothing here touches global RNG state or the clock.

All functions operate on plain lists of floats and return plain floats/dicts so
they are trivially unit-testable (see ``analyze.py --self-test``).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Order statistics
# ---------------------------------------------------------------------------
def median(xs: list[float]) -> float:
    """Median with linear interpolation for even-length samples."""
    if not xs:
        raise ValueError("median of empty sample")
    return quantile(xs, 0.5)


def quantile(xs: list[float], q: float) -> float:
    """The ``q``-quantile (0..1) using the type-7 / linear-interpolation rule.

    This matches numpy's default ``np.quantile`` so results are comparable if
    someone re-runs the numbers with numpy later.
    """
    if not xs:
        raise ValueError("quantile of empty sample")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"quantile q must be in [0,1], got {q}")
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    pos = q * (len(s) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(s[lo])
    frac = pos - lo
    return float(s[lo] * (1.0 - frac) + s[hi] * frac)


def mean(xs: list[float]) -> float:
    if not xs:
        raise ValueError("mean of empty sample")
    return sum(xs) / len(xs)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
@dataclass
class CI:
    point: float
    low: float
    high: float
    level: float  # e.g. 0.95
    n: int        # sample size the CI was computed from
    n_boot: int

    def as_dict(self) -> dict:
        return {
            "point": self.point,
            "ci_low": self.low,
            "ci_high": self.high,
            "ci_level": self.level,
            "n": self.n,
            "n_boot": self.n_boot,
        }

    def straddles(self, value: float) -> bool:
        """True if ``value`` lies inside [low, high] (inclusive)."""
        return self.low <= value <= self.high


def _percentile_ci(boot_stats: list[float], point: float, level: float,
                   n: int, n_boot: int) -> CI:
    alpha = (1.0 - level) / 2.0
    low = quantile(boot_stats, alpha)
    high = quantile(boot_stats, 1.0 - alpha)
    return CI(point=point, low=low, high=high, level=level, n=n, n_boot=n_boot)


def bootstrap_ci(values: list[float], statistic, *, n_boot: int, level: float,
                 seed: int) -> CI:
    """Percentile bootstrap CI for ``statistic`` over a 1-D sample.

    ``statistic`` maps a resample (list[float]) -> float. Resampling is with
    replacement over the sample indices, using a private seeded RNG so the CI
    is byte-for-byte reproducible.
    """
    if not values:
        raise ValueError("bootstrap of empty sample")
    point = statistic(values)
    if len(values) == 1:
        # No variability to resample — degenerate CI at the point estimate.
        return CI(point=point, low=point, high=point, level=level, n=1, n_boot=0)
    rng = random.Random(seed)
    n = len(values)
    boot = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot.append(statistic(sample))
    return _percentile_ci(boot, point, level, n, n_boot)


def paired_bootstrap_ratio_ci(pairs: list[tuple[float, float]], *,
                              n_boot: int, level: float, seed: int) -> CI:
    """Paired percentile bootstrap for the **median of per-pair ratios** b/a.

    ``pairs`` is a list of (control_value, treatment_value) tuples — one per
    task (the exchangeable unit of analysis). We resample *pairs* with
    replacement (preserving the pairing) and recompute median(b/a). This is the
    per-task token-ratio effect size the plan pre-registers.

    Pairs whose control value is <= 0 are dropped (an undefined ratio); if that
    empties the sample we raise, because a silent empty CI would be misleading.
    """
    usable = [(a, b) for (a, b) in pairs if a and a > 0]
    if not usable:
        raise ValueError("no usable pairs for ratio CI (all control values <= 0)")

    def stat(sample: list[tuple[float, float]]) -> float:
        return median([b / a for (a, b) in sample])

    point = stat(usable)
    if len(usable) == 1:
        return CI(point=point, low=point, high=point, level=level, n=1, n_boot=0)
    rng = random.Random(seed)
    n = len(usable)
    boot = []
    for _ in range(n_boot):
        resample = [usable[rng.randrange(n)] for _ in range(n)]
        boot.append(stat(resample))
    return _percentile_ci(boot, point, level, n, n_boot)


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test (paired, non-parametric)
# ---------------------------------------------------------------------------
@dataclass
class WilcoxonResult:
    statistic: float      # W = sum of positive-signed ranks
    p_value: float        # two-sided
    n: int                # number of non-zero-difference pairs used
    n_zeros: int          # dropped zero-difference pairs
    method: str           # "exact" | "normal-approx"
    rank_biserial: float  # effect size in [-1, 1]

    def as_dict(self) -> dict:
        return {
            "statistic_W": self.statistic,
            "p_value": self.p_value,
            "n_pairs": self.n,
            "n_zero_diffs": self.n_zeros,
            "method": self.method,
            "rank_biserial": self.rank_biserial,
        }


def _rankdata(values: list[float]) -> list[float]:
    """Average-rank of ``values`` (1-based), assigning tied ranks their mean."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # average of positions i..j, converted to 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _exact_two_sided_p(w_plus: float, n: int) -> float:
    """Exact two-sided p-value for W+ under H0 by full enumeration of 2^n signs.

    Only called for small ``n`` (<= EXACT_MAX in wilcoxon_signed_rank), so the
    2^n enumeration is cheap. Assumes no ties in |diff| (caller guarantees the
    exact branch is only used when there are none).
    """
    ranks = list(range(1, n + 1))
    total = 1 << n
    # Distribution of W+ over all sign assignments.
    from collections import defaultdict
    dist: dict[float, int] = defaultdict(int)
    for mask in range(total):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += ranks[i]
        dist[s] += 1
    mean_w = n * (n + 1) / 4.0
    # Two-sided: probability of a |W+ - mean| at least as extreme as observed.
    obs_dev = abs(w_plus - mean_w)
    tail = sum(c for w, c in dist.items() if abs(w - mean_w) >= obs_dev - 1e-9)
    return min(1.0, tail / total)


def _normal_approx_two_sided_p(w_plus: float, n: int,
                               tie_correction: float) -> float:
    """Normal approximation with continuity correction and tie correction."""
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0 - tie_correction
    if var_w <= 0:
        return 1.0
    # Continuity correction toward the mean.
    z = (abs(w_plus - mean_w) - 0.5) / math.sqrt(var_w)
    if z < 0:
        z = 0.0
    # Two-sided p = 2 * (1 - Phi(z)); erfc gives the upper tail directly.
    return min(1.0, math.erfc(z / math.sqrt(2.0)))


# Below this many non-zero pairs we enumerate the exact null distribution.
EXACT_MAX = 18


def wilcoxon_signed_rank(x: list[float], y: list[float]) -> WilcoxonResult:
    """Two-sided Wilcoxon signed-rank test on paired samples ``x`` (control) and
    ``y`` (treatment).

    Returns W+ (sum of ranks of positive differences y-x), a two-sided p-value
    (exact for small n and no ties, otherwise a tie-corrected normal
    approximation), and the rank-biserial correlation as a standardized effect
    size. Zero differences are dropped (Wilcoxon's standard handling).

    With the tiny task counts of the thin slice this test has limited power; the
    caller is expected to surface n and the exact/approx flag so readers can
    judge the evidence honestly rather than over-reading a p-value.
    """
    if len(x) != len(y):
        raise ValueError("wilcoxon: x and y must be the same length")
    diffs = [b - a for a, b in zip(x, y)]
    nonzero = [d for d in diffs if d != 0]
    n_zeros = len(diffs) - len(nonzero)
    n = len(nonzero)
    if n == 0:
        return WilcoxonResult(statistic=0.0, p_value=1.0, n=0, n_zeros=n_zeros,
                              method="degenerate", rank_biserial=0.0)

    abs_ranks = _rankdata([abs(d) for d in nonzero])
    w_plus = sum(r for d, r in zip(nonzero, abs_ranks) if d > 0)
    w_minus = sum(r for d, r in zip(nonzero, abs_ranks) if d < 0)
    total_rank = n * (n + 1) / 2.0
    # Rank-biserial correlation: (W+ - W-) / total. +1 => treatment strictly
    # larger on every pair, -1 => strictly smaller.
    rank_biserial = (w_plus - w_minus) / total_rank if total_rank else 0.0

    # Detect ties in |diff| — they invalidate the exact enumeration branch.
    abs_vals = [abs(d) for d in nonzero]
    has_ties = len(set(abs_vals)) != len(abs_vals)

    if n <= EXACT_MAX and not has_ties:
        p = _exact_two_sided_p(w_plus, n)
        method = "exact"
    else:
        # Tie correction term: sum(t^3 - t)/48 over tie groups of size t.
        from collections import Counter
        counts = Counter(abs_vals)
        tie_corr = sum((t ** 3 - t) for t in counts.values() if t > 1) / 48.0
        p = _normal_approx_two_sided_p(w_plus, n, tie_corr)
        method = "normal-approx"

    return WilcoxonResult(statistic=w_plus, p_value=p, n=n, n_zeros=n_zeros,
                          method=method, rank_biserial=rank_biserial)


# ---------------------------------------------------------------------------
# Proportion CI (Wilson score) — for success / hallucination rates
# ---------------------------------------------------------------------------
def wilson_ci(successes: int, total: int, level: float = 0.95) -> CI:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal (Wald) interval for the small n and near-0/near-1
    rates we expect from task-success counts, where Wald intervals misbehave.
    """
    if total == 0:
        return CI(point=float("nan"), low=float("nan"), high=float("nan"),
                  level=level, n=0, n_boot=0)
    p = successes / total
    # z for a two-sided ``level`` interval via the inverse error function.
    z = math.sqrt(2.0) * _erfinv(level)
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return CI(point=p, low=max(0.0, center - half), high=min(1.0, center + half),
              level=level, n=total, n_boot=0)


def _erfinv(x: float) -> float:
    """Inverse error function of a *confidence level* argument.

    ``wilson_ci`` needs the z matching a two-sided interval of mass ``level``:
    z = sqrt(2) * erfinv(level). We implement erfinv via a short Newton
    refinement on erf (available in math), which is ample precision for CIs.
    """
    if not -1.0 < x < 1.0:
        raise ValueError("erfinv domain is (-1, 1)")
    # Winitzki approximation as the initial guess.
    a = 0.147
    ln = math.log(1 - x * x)
    t1 = 2 / (math.pi * a) + ln / 2
    guess = math.copysign(math.sqrt(math.sqrt(t1 * t1 - ln / a) - t1), x)
    # Two Newton steps against math.erf.
    for _ in range(2):
        err = math.erf(guess) - x
        deriv = (2.0 / math.sqrt(math.pi)) * math.exp(-guess * guess)
        guess -= err / deriv
    return guess
