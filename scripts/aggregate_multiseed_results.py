#!/usr/bin/env python3
"""
Aggregate multi-seed experiment results, compute variance bounds (95% CI),
and perform paired statistical significance testing.
Addresses Problem P2-4.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np

from utils.stats_analyzer import (
    compute_summary_stats,
    compute_paired_comparison,
    format_metric_with_ci,
    generate_variance_markdown_table,
)


def aggregate_results(results_dir: Path, metric_key: str = "metrics/mAP50(M)") -> Dict[str, List[float]]:
    """
    Scan a results directory for experiment JSON files and group metrics by experiment base name.
    """
    grouped_data: Dict[str, List[float]] = {}

    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Determine experiment base name by stripping seed suffix
            stem = json_file.stem
            val = None
            if metric_key in data:
                val = float(data[metric_key])
            elif "metrics/mask_mAP50" in data:
                val = float(data["metrics/mask_mAP50"])
            elif "mask_mAP50" in data:
                val = float(data["mask_mAP50"])
            elif "metrics/mAP50(M)" in data:
                val = float(data["metrics/mAP50(M)"])
            elif "direct_mask_mAP50" in data:
                val = float(data["direct_mask_mAP50"])
            elif "In-Domain Mask mAP50" in data:
                val = float(data["In-Domain Mask mAP50"])

            if val is not None:
                # Group by base experiment name
                base_name = stem
                for seed_str in ["_seed42", "_seed123", "_seed456", "_seed0"]:
                    if seed_str in base_name:
                        base_name = base_name.replace(seed_str, "")
                        break

                grouped_data.setdefault(base_name, []).append(val)
        except Exception:
            continue

    return grouped_data


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed experimental results.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing experiment JSON outputs")
    parser.add_argument("--metric", type=str, default="metrics/mAP50(M)", help="Metric key to analyze")
    parser.add_argument("--baseline", type=str, default="baseline_finetune_clean", help="Baseline arm name for comparison")
    parser.add_argument("--treatment", type=str, default="prod_mask_kd_box_only", help="Treatment arm name for comparison")
    args = parser.parse_args()

    res_dir = Path(args.results_dir)
    if not res_dir.exists():
        print(f"Results directory '{res_dir}' does not exist.")
        return

    grouped = aggregate_results(res_dir, metric_key=args.metric)
    if not grouped:
        print(f"No experiment JSON files found with metric '{args.metric}' in '{res_dir}'.")
        return

    print("=" * 70)
    print(f"MULTI-SEED VARIANCE BOUNDS & STATISTICAL ANALYSIS ({args.metric})")
    print("=" * 70)
    print(generate_variance_markdown_table(grouped, metric_name=args.metric))
    print("=" * 70)

    # Perform paired comparison if both baseline and treatment exist with equal seed counts
    if args.baseline in grouped and args.treatment in grouped:
        b_vals = grouped[args.baseline]
        t_vals = grouped[args.treatment]
        if len(b_vals) == len(t_vals) and len(b_vals) >= 2:
            comp = compute_paired_comparison(b_vals, t_vals)
            print("\n🔬 PAIRED HYPOTHESIS TESTING (Treatment vs Baseline):")
            print(f"  Seeds Tested       : {comp['n_seeds']}")
            print(f"  Mean Delta         : +{comp['delta_mean']:.4f} (Std: ±{comp['delta_std']:.4f})")
            print(f"  Effect Size (d)    : {comp['cohens_d']:.2f}")
            print(f"  Paired t-test p    : {comp['p_value_ttest']:.4e}")
            print(f"  Wilcoxon p         : {comp['p_value_wilcoxon']:.4e}")
            print(f"  Statistically Sig  : {'YES (p < 0.05)' if comp['statistically_significant_05'] else 'NO (p >= 0.05)'}")
            print("=" * 70)


if __name__ == "__main__":
    main()
