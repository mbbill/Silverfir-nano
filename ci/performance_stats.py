"""Shared probability model for performance benchmark providers."""

from __future__ import annotations

import math
from typing import Any


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean requires positive samples")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the continued fraction used by regularized incomplete beta."""
    max_iterations = 200
    epsilon = 3e-14
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d

    for iteration in range(1, max_iterations + 1):
        even = 2 * iteration
        coefficient = (
            iteration * (b - iteration) * x
            / ((qam + even) * (a + even))
        )
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c

        coefficient = -(
            (a + iteration) * (qab + iteration) * x
            / ((a + even) * (qap + even))
        )
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= epsilon:
            return result

    raise ArithmeticError("incomplete beta continued fraction did not converge")


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        raise ValueError("beta parameters must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    log_factor = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    factor = math.exp(log_factor)
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * _beta_continued_fraction(a, b, x) / a
    return 1.0 - (
        factor * _beta_continued_fraction(b, a, 1.0 - x) / b
    )


def student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    """CDF of Student's t without a scipy dependency."""
    if degrees_of_freedom <= 0:
        raise ValueError("degrees of freedom must be positive")
    if value == 0.0:
        return 0.5
    if math.isinf(value):
        return 1.0 if value > 0.0 else 0.0

    df = float(degrees_of_freedom)
    x = df / (df + value * value)
    tail = 0.5 * regularized_incomplete_beta(x, df / 2.0, 0.5)
    return 1.0 - tail if value > 0.0 else tail


def probability_summary(pair_ratios: list[float]) -> dict[str, Any]:
    if len(pair_ratios) < 2:
        raise ValueError("probability analysis requires at least two pairs")
    if any(ratio <= 0.0 for ratio in pair_ratios):
        raise ValueError("paired ratios must be positive")

    log_deltas = [math.log(ratio) for ratio in pair_ratios]
    pair_count = len(log_deltas)
    mean_log = math.fsum(log_deltas) / pair_count
    squared = math.fsum(
        (value - mean_log) ** 2 for value in log_deltas
    )
    stddev_log = math.sqrt(squared / (pair_count - 1))
    standard_error = stddev_log / math.sqrt(pair_count)

    if standard_error == 0.0:
        if mean_log < 0.0:
            probability_regression = 1.0
        elif mean_log > 0.0:
            probability_regression = 0.0
        else:
            probability_regression = 0.5
    else:
        zero_t = -mean_log / standard_error
        probability_regression = student_t_cdf(
            zero_t, pair_count - 1
        )

    return {
        "pair_count": pair_count,
        "log_deltas": log_deltas,
        "mean_log_ratio": mean_log,
        "log_ratio_stddev": stddev_log,
        "standard_error": standard_error,
        "probability_regression": probability_regression,
        "probability_improvement": 1.0 - probability_regression,
        "delta_percent": math.expm1(mean_log) * 100.0,
        "volatility_percent": math.expm1(stddev_log) * 100.0,
        "pair_deltas_percent": [
            math.expm1(value) * 100.0 for value in log_deltas
        ],
    }


def direction_probability_for_pairs(
    *,
    mean_log_ratio: float,
    log_ratio_stddev: float,
    pair_count: int,
) -> float:
    if pair_count < 2:
        raise ValueError("pair count must be at least two")
    magnitude = abs(mean_log_ratio)
    if magnitude == 0.0:
        return 0.5
    if log_ratio_stddev == 0.0:
        return 1.0
    t_value = magnitude / (log_ratio_stddev / math.sqrt(pair_count))
    return student_t_cdf(t_value, pair_count - 1)


def effect_exceeds_probability(
    *,
    mean_log_ratio: float,
    log_ratio_stddev: float,
    pair_count: int,
    minimum_effect_log: float,
) -> float:
    """Probability that the true effect magnitude exceeds the floor.

    Shifts the paired t statistic by the practical-significance floor, so
    with a zero floor this reduces to the plain directional probability.
    """
    if pair_count < 2:
        raise ValueError("pair count must be at least two")
    if minimum_effect_log < 0.0:
        raise ValueError("minimum effect must not be negative")
    magnitude = abs(mean_log_ratio) - minimum_effect_log
    if log_ratio_stddev == 0.0:
        return 1.0 if magnitude > 0.0 else 0.0
    t_value = magnitude / (log_ratio_stddev / math.sqrt(pair_count))
    return student_t_cdf(t_value, pair_count - 1)


def family_adjusted_probability(
    probability: float,
    *,
    metric_count: int,
    job_count: int,
    maximum_looks: int,
) -> float:
    """Convert a family-wide probability into a per-look threshold."""
    if not 0.5 < probability < 1.0:
        raise ValueError("probability must be between 0.5 and 1")
    if metric_count <= 0:
        raise ValueError("metric count must be positive")
    if job_count <= 0:
        raise ValueError("job count must be positive")
    if maximum_looks <= 0:
        raise ValueError("maximum looks must be positive")
    hypotheses = metric_count * job_count * maximum_looks
    return 1.0 - (1.0 - probability) / hypotheses


def required_pairs(
    metric: dict[str, Any],
    *,
    probability: float,
    minimum_pairs: int,
    maximum_pairs: int,
) -> int | None:
    """Predict a fixed final pair count from the initial effect and variance."""
    if not 0.5 < probability < 1.0:
        raise ValueError("probability must be between 0.5 and 1")
    current_pairs = int(metric["pair_count"])
    start = max(current_pairs, minimum_pairs)
    if start % 2:
        start += 1

    for pair_count in range(start, maximum_pairs + 1, 2):
        observed_probability = direction_probability_for_pairs(
            mean_log_ratio=float(metric["mean_log_ratio"]),
            log_ratio_stddev=float(metric["log_ratio_stddev"]),
            pair_count=pair_count,
        )
        if observed_probability >= probability:
            return pair_count
    return None


def required_pairs_for_direction(
    metric: dict[str, Any],
    *,
    direction: str,
    probability: float,
    minimum_pairs: int,
    maximum_pairs: int,
) -> int | None:
    """Predict pairs for a direction frozen before this sample was taken."""
    mean_log = float(metric["mean_log_ratio"])
    if direction == "regression":
        directed_mean = -mean_log
    elif direction == "improvement":
        directed_mean = mean_log
    else:
        raise ValueError(f"unknown candidate direction: {direction}")

    if directed_mean <= 0.0:
        return None
    return required_pairs(
        metric,
        probability=probability,
        minimum_pairs=minimum_pairs,
        maximum_pairs=maximum_pairs,
    )


def metric_plans(
    metrics: dict[str, dict[str, Any]],
    *,
    regression_probability: float,
    improvement_probability: float,
    minimum_pairs: int,
    maximum_pairs: int,
    pilot_probability: float = 0.8,
) -> dict[str, dict[str, Any]]:
    if not 0.5 < pilot_probability < 1.0:
        raise ValueError("pilot probability must be between 0.5 and 1")
    plans: dict[str, dict[str, Any]] = {}
    for name, metric in metrics.items():
        mean_log = float(metric["mean_log_ratio"])
        if mean_log < 0.0:
            direction = "regression"
            target_probability = regression_probability
            pilot_direction_probability = float(
                metric["probability_regression"]
            )
        elif mean_log > 0.0:
            direction = "improvement"
            target_probability = improvement_probability
            pilot_direction_probability = float(
                metric["probability_improvement"]
            )
        else:
            direction = None
            target_probability = None
            pilot_direction_probability = 0.5

        target_pairs = None
        if direction is not None:
            target_pairs = required_pairs(
                metric,
                probability=float(target_probability),
                minimum_pairs=minimum_pairs,
                maximum_pairs=maximum_pairs,
            )
        plans[name] = {
            "direction": direction,
            "target_probability": target_probability,
            "target_pairs": target_pairs,
            "pilot_direction_probability": pilot_direction_probability,
            "selected": (
                direction is not None
                and pilot_direction_probability >= pilot_probability
            ),
        }
    return plans


def classify_metrics(
    *,
    initial: dict[str, dict[str, Any]],
    final: dict[str, dict[str, Any]],
    plans: dict[str, dict[str, Any]],
    regression_probability: float,
    improvement_probability: float,
    identical_binaries: bool = False,
    minimum_effect_log: float = 0.0,
    placement_metrics: frozenset[str] = frozenset(),
) -> dict[str, dict[str, Any]]:
    """Classify only directions frozen by the initial screening stage.

    A regression additionally needs the data to support a true effect
    beyond ``minimum_effect_log`` at the same confidence; a directional
    hit inside the floor reports NEGLIGIBLE and does not fail the gate.
    """
    metrics: dict[str, dict[str, Any]] = {}
    for name, initial_metric in initial.items():
        plan = plans[name]
        selected = bool(plan["selected"])
        measured = final.get(name) if selected else initial_metric
        if measured is None:
            raise ValueError(f"{name}: selected metric has no final sample")

        direction = plan["direction"]
        if direction == "regression" and selected:
            if measured["probability_regression"] < regression_probability:
                status = "RECOVERED"
            elif (
                minimum_effect_log > 0.0
                and float(measured["mean_log_ratio"]) < 0.0
                and effect_exceeds_probability(
                    mean_log_ratio=float(measured["mean_log_ratio"]),
                    log_ratio_stddev=float(measured["log_ratio_stddev"]),
                    pair_count=int(measured["pair_count"]),
                    minimum_effect_log=minimum_effect_log,
                )
                < regression_probability
            ):
                status = "NEGLIGIBLE"
            elif name in placement_metrics:
                # Evidence (mcts_mem interpreter/dispatch.md, 2026-08-01):
                # these rows measure the binary's address draw, not its
                # code. Identical sources with identical dispatch counts
                # swing them by whole percent across builds and runners.
                # Interim classification until the gate can verify a
                # confirmed row against a layout-perturbed rebuild.
                status = "PLACEMENT"
            else:
                status = "REGRESSION"
        elif direction == "improvement" and selected:
            status = (
                "IMPROVEMENT"
                if measured["probability_improvement"]
                >= improvement_probability
                else "PASS"
            )
        else:
            status = "PASS"

        # Byte-identical executables cannot contain a source performance
        # change. Keep an apparent regression visible as runner drift, but
        # never let it fail the gate or claim a source improvement.
        if identical_binaries:
            if status == "REGRESSION":
                status = "UNSTABLE"
            elif status == "IMPROVEMENT":
                status = "PASS"

        metrics[name] = {
            **measured,
            "initial": initial_metric,
            "candidate_direction": direction,
            "pilot_direction_probability": plan[
                "pilot_direction_probability"
            ],
            "selected": selected,
            "target_pairs": plan["target_pairs"],
            "status": status,
        }
    return metrics
