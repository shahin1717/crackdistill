"""
Statistical analysis utility for multi-seed validation and variance bounds.
Addresses Problem P2-4:
- Computes sample mean, sample standard deviation, and standard error (SE).
- Computes 95% Confidence Intervals via Student's t-distribution.
- Performs paired Student's t-tests and Wilcoxon signed-rank tests for statistical significance.
- Computes standardized effect size (Cohen's d).
- Formats LaTeX and Markdown comparative tables with error bounds (mean ± 95% CI).
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def compute_summary_stats(values: Union[List[float], np.ndarray], confidence: float = 0.95) -> Dict[str, float]:
    """
    Compute sample mean, standard deviation, standard error, and confidence intervals.

    Args:
        values (List[float] | np.ndarray): Sample values across multiple random seeds (N >= 2).
        confidence (float): Confidence level (default 0.95 for 95% CI).

    Returns:
        Dict[str, float]: Dictionary containing n, mean, std, se, ci_margin, ci_lower, ci_upper.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        raise ValueError("Cannot compute statistics on an empty array.")

    mean = float(np.mean(arr))
    if n == 1:
        return {
            "n": 1,
            "mean": mean,
            "std": 0.0,
            "se": 0.0,
            "ci_margin": 0.0,
            "ci_lower": mean,
            "ci_upper": mean,
        }

    # Unbiased sample standard deviation (ddof=1)
    std = float(np.std(arr, ddof=1))
    se = std / math.sqrt(n)

    # Student's t critical value
    try:
        from scipy import stats
        t_crit = float(stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1))
    except ImportError:
        # Standard lookup table for common sample sizes at 95% confidence
        t_table_95 = {
            1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
            6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
            20: 2.086, 30: 2.042
        }
        df = n - 1
        t_crit = t_table_95.get(df, 1.96 if df > 30 else 2.5)

    ci_margin = t_crit * se

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "se": se,
        "ci_margin": ci_margin,
        "ci_lower": mean - ci_margin,
        "ci_upper": mean + ci_margin,
    }


def compute_paired_comparison(
    baseline_values: Union[List[float], np.ndarray],
    treatment_values: Union[List[float], np.ndarray],
) -> Dict[str, Any]:
    """
    Perform paired comparison between baseline control runs and treatment KD runs across identical seeds.

    Args:
        baseline_values: Metrics for baseline runs indexed by seed.
        treatment_values: Metrics for treatment runs indexed by the same seeds.

    Returns:
        Dict[str, Any]: Delta metrics, p-values (paired t-test and Wilcoxon), and Cohen's d.
    """
    b = np.asarray(baseline_values, dtype=np.float64)
    t = np.asarray(treatment_values, dtype=np.float64)

    if len(b) != len(t):
        raise ValueError(f"Sample size mismatch: baseline has {len(b)} seeds, treatment has {len(t)} seeds.")

    n = len(b)
    diff = t - b
    delta_mean = float(np.mean(diff))

    b_stats = compute_summary_stats(b)
    t_stats = compute_summary_stats(t)

    # Cohen's d for paired samples
    diff_std = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    cohen_d = (delta_mean / diff_std) if diff_std > 1e-9 else 0.0

    p_value_ttest = 1.0
    p_value_wilcoxon = 1.0

    if n >= 2:
        try:
            from scipy import stats
            # Paired t-test
            t_res = stats.ttest_rel(t, b)
            p_value_ttest = float(t_res.pvalue)

            # Wilcoxon signed-rank test (if n >= 3 and not all diffs are identical)
            if n >= 3 and not np.all(diff == diff[0]):
                w_res = stats.wilcoxon(t, b)
                p_value_wilcoxon = float(w_res.pvalue)
        except Exception:
            pass

    return {
        "n_seeds": n,
        "baseline": b_stats,
        "treatment": t_stats,
        "delta_mean": delta_mean,
        "delta_std": diff_std,
        "cohens_d": cohen_d,
        "p_value_ttest": p_value_ttest,
        "p_value_wilcoxon": p_value_wilcoxon,
        "statistically_significant_05": p_value_ttest < 0.05,
    }


def format_metric_with_ci(stats_dict: Dict[str, float], precision: int = 4) -> str:
    """Format metric as 'mean ± ci_margin' string."""
    fmt = f"{{:.{precision}f}}"
    m_str = fmt.format(stats_dict["mean"])
    ci_str = fmt.format(stats_dict["ci_margin"])
    return f"{m_str} ± {ci_str}"


def generate_variance_markdown_table(
    experiments_data: Dict[str, List[float]],
    metric_name: str = "Mask mAP50",
    precision: int = 4,
) -> str:
    """Generate Markdown summary table for multi-seed experimental results."""
    lines = [
        f"| Experiment Arm | N Seeds | Mean {metric_name} | Std Dev | 95% Confidence Interval |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    for exp_name, values in experiments_data.items():
        st = compute_summary_stats(values)
        fmt = f"{{:.{precision}f}}"
        mean_str = fmt.format(st["mean"])
        std_str = fmt.format(st["std"])
        ci_str = f"[{fmt.format(st['ci_lower'])}, {fmt.format(st['ci_upper'])}]"
        lines.append(f"| **{exp_name}** | {st['n']} | {mean_str} | ±{std_str} | {ci_str} |")

    return "\n".join(lines)


# Aliases and helper wrappers for flexible usage across scripts & tests
compute_stats = compute_summary_stats


def compare_two_arms(
    arm_a: Union[List[float], np.ndarray],
    arm_b: Union[List[float], np.ndarray],
    paired: bool = True
) -> Dict[str, Any]:
    """Compare two experimental arms with paired difference and effect size."""
    res = compute_paired_comparison(arm_a, arm_b)
    diff = np.asarray(arm_b) - np.asarray(arm_a)
    diff_stats = compute_summary_stats(diff)
    return {
        "n": res["n_seeds"],
        "diff_mean": res["delta_mean"],
        "diff_std": res["delta_std"],
        "cohens_d": res["cohens_d"],
        "t_pvalue": res["p_value_ttest"],
        "wilcoxon_pvalue": res["p_value_wilcoxon"],
        "ci_lower": diff_stats["ci_lower"],
        "ci_upper": diff_stats["ci_upper"],
        "is_significant": res["statistically_significant_05"],
    }


def format_comparison_table(
    arm_a: Union[List[float], np.ndarray],
    arm_b: Union[List[float], np.ndarray],
    arm_a_name: str = "Arm A",
    arm_b_name: str = "Arm B",
    metric_name: str = "Mask mAP50",
) -> str:
    """Format comparative table between two arms."""
    comp = compare_two_arms(arm_a, arm_b)
    st_a = compute_summary_stats(arm_a)
    st_b = compute_summary_stats(arm_b)
    lines = [
        f"| Arm | N Seeds | Mean ± Std | 95% CI |",
        f"| :--- | :---: | :---: | :---: |",
        f"| {arm_a_name} | {st_a['n']} | {st_a['mean']:.4f} ± {st_a['std']:.4f} | [{st_a['ci_lower']:.4f}, {st_a['ci_upper']:.4f}] |",
        f"| {arm_b_name} | {st_b['n']} | {st_b['mean']:.4f} ± {st_b['std']:.4f} | [{st_b['ci_lower']:.4f}, {st_b['ci_upper']:.4f}] |",
        f"| **Delta ({arm_b_name} - {arm_a_name})** | {comp['n']} | **{comp['diff_mean']:+.4f}** (Cohen's d: {comp['cohens_d']:.2f}) | [{comp['ci_lower']:.4f}, {comp['ci_upper']:.4f}] |",
    ]
    return "\n".join(lines)

