"""The one statistic the exam leans on: a lower bound for accuracy."""

from math import sqrt
from statistics import NormalDist


def wilson_lower_bound(successes: int, n: int, confidence: float = 0.90) -> float:
    """One-sided Wilson score lower bound for a proportion.

    With confidence 0.90 we can say: there is a 90% chance the true accuracy is at least this.
    On 20 cases with 19 right the point estimate is 95%, the bound is about 85%. A small exam
    cannot claim a high accuracy, which is the point.
    """
    if n <= 0:
        return 0.0
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    if not 0.5 < confidence < 1:
        raise ValueError("confidence must be between 0.5 and 1")
    z = NormalDist().inv_cdf(confidence)
    p = successes / n
    z2 = z * z
    centre = p + z2 / (2 * n)
    margin = z * sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, (centre - margin) / (1 + z2 / n))
