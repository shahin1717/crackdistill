#!/usr/bin/env python3
"""
Aggregate per-seed results into arm-level statistics and seed-paired comparisons.
Addresses Problem P2-4.

Reads every JSON in --results-dir that carries "arm" and "seed" (written by the training notebooks'
Step 4 and by scripts/evaluate_canonical_test_set.py --arm/--seed). The metric may be top-level or
nested (e.g. metrics_indomain.mask_mAP50_95). Runs are paired by seed, never by file order.

Usage:
  python scripts/aggregate_multiseed_results.py --results-dir results/test --metric in_domain_mask_mAP50
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.stats_analyzer import (
    compute_paired_comparison,
    format_metric_with_ci,
    compute_summary_stats,
    generate_variance_markdown_table,
)


def find_metric(data: dict, key: str) -> Optional[float]:
    """Return data[key], or the value of key inside any nested dict (one level deep)."""
    if key in data and isinstance(data[key], (int, float)):
        return float(data[key])
    for v in data.values():
        if isinstance(v, dict) and isinstance(v.get(key), (int, float)):
            return float(v[key])
    return None


def load_runs(results_dir: Path, metric_key: str) -> Dict[str, Dict[int, float]]:
    """{arm: {seed: value}}. Two files for the same (arm, seed) are an error, not a silent overwrite."""
    runs: Dict[str, Dict[int, float]] = {}
    sources: Dict[tuple, Path] = {}
    for json_file in sorted(Path(results_dir).glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or "arm" not in data or "seed" not in data:
            continue
        value = find_metric(data, metric_key)
        if value is None:
            continue
        arm, seed = str(data["arm"]), int(data["seed"])
        if (arm, seed) in sources:
            raise ValueError(f"Duplicate result for arm={arm} seed={seed}: {sources[(arm, seed)]} and {json_file}")
        sources[(arm, seed)] = json_file
        runs.setdefault(arm, {})[seed] = value
    return runs


def aggregate_results(results_dir: Path, metric_key: str = "mask_mAP50_95") -> Dict[str, List[float]]:
    """{arm: [values ordered by seed]}."""
    return {arm: [by_seed[s] for s in sorted(by_seed)] for arm, by_seed in load_runs(results_dir, metric_key).items()}


def paired_by_seed(runs: Dict[str, Dict[int, float]], baseline: str, treatment: str):
    """Seeds present in both arms, with the matching values."""
    seeds = sorted(set(runs.get(baseline, {})) & set(runs.get(treatment, {})))
    return seeds, [runs[baseline][s] for s in seeds], [runs[treatment][s] for s in seeds]


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed experimental results.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing per-run JSON files")
    parser.add_argument("--metric", type=str, default="in_domain_mask_mAP50", help="Metric key (top-level or nested); primary = test mask mAP50")
    parser.add_argument("--baseline", type=str, default="baseline", help="Baseline arm name")
    parser.add_argument("--treatment", type=str, default="mask_kd", help="Treatment arm name")
    parser.add_argument("--control", type=str, default="gt_soft", help="Optional control arm compared against the treatment")
    args = parser.parse_args()

    runs = load_runs(Path(args.results_dir), args.metric)
    if not runs:
        print(f"No JSON files with 'arm', 'seed' and metric '{args.metric}' found in '{args.results_dir}'.")
        return 1

    print("=" * 70)
    print(f"MULTI-SEED SUMMARY ({args.metric})")
    print("=" * 70)
    print(generate_variance_markdown_table({a: [v[s] for s in sorted(v)] for a, v in runs.items()}, metric_name=args.metric))
    for arm, by_seed in runs.items():
        print(f"  {arm}: seeds {sorted(by_seed)}")

    for ref, other in [(args.baseline, args.treatment), (args.control, args.treatment)]:
        seeds, ref_vals, other_vals = paired_by_seed(runs, ref, other)
        if len(seeds) < 2:
            if ref in runs and other in runs:
                print(f"\n[{other} vs {ref}] fewer than 2 shared seeds ({seeds}); no paired test.")
            continue
        comp = compute_paired_comparison(ref_vals, other_vals)
        print(f"\n🔬 PAIRED BY SEED ({other} vs {ref}), seeds {seeds}:")
        print(f"  {ref:<12}: {format_metric_with_ci(comp['baseline'])}")
        print(f"  {other:<12}: {format_metric_with_ci(comp['treatment'])}")
        diff_stats = compute_summary_stats([o - r for r, o in zip(ref_vals, other_vals)])
        print(f"  Δ mean     : {comp['delta_mean']:+.4f}  (95% CI [{diff_stats['ci_lower']:+.4f}, {diff_stats['ci_upper']:+.4f}])")
        print(f"  Cohen's d  : {comp['cohens_d']:.3f}")
        print(f"  paired t   : p = {comp['p_value_ttest']:.4f}   Wilcoxon: p = {comp['p_value_wilcoxon']:.4f}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
