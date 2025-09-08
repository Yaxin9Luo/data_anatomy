#!/usr/bin/env python3
"""
Benchmark Evaluation Tool for Label-Shift Estimation Methods

This tool provides standardized evaluation metrics and comparison capabilities
for different approaches to estimating training data mixture proportions from
language model generations.

Usage:
    python benchmark_evaluation.py --results_dir out/labelshift_llama7b --ground_truth bench/specs/llama-1.yaml
    python benchmark_evaluation.py --compare out/method1 out/method2 --ground_truth bench/specs/llama-1.yaml
"""

import argparse
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import pandas as pd


@dataclass
class MethodResult:
    """Container for a method's results"""
    name: str
    estimates: Dict[str, float]
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None
    metadata: Optional[Dict] = None


@dataclass
class BenchmarkMetrics:
    """Container for benchmark evaluation metrics"""
    l1_distance: float
    l2_distance: float
    kl_divergence: float
    max_absolute_error: float
    mean_absolute_error: float
    relative_errors: Dict[str, float]
    coverage_rate: Optional[float] = None  # For methods with confidence intervals
    interval_widths: Optional[Dict[str, float]] = None


class LabelShiftBenchmark:
    """Main benchmark evaluation class"""
    
    def __init__(self, ground_truth_file: str):
        """
        Initialize benchmark with ground truth mixture proportions.
        
        Args:
            ground_truth_file: Path to YAML file with true category weights
        """
        self.ground_truth_file = ground_truth_file
        self.ground_truth = self._load_ground_truth(ground_truth_file)
        self.categories = list(self.ground_truth.keys())
        
    def _load_ground_truth(self, filepath: str) -> Dict[str, float]:
        """Load ground truth proportions from YAML specification"""
        with open(filepath, 'r') as f:
            spec = yaml.safe_load(f)
        
        weights = spec['category_weights']
        # Normalize to probabilities
        total = sum(weights.values())
        return {cat: weight/total for cat, weight in weights.items()}
    
    def _load_method_results(self, results_dir: str) -> MethodResult:
        """
        Load results from a method's output directory.
        
        Expected format: method should provide results for all 7 categories:
        - CommonCrawl, C4, GitHub, Wikipedia, Books, Arxiv, StackExchange
        
        Alternative category names are mapped automatically:
        - Code -> GitHub
        - Papers -> Arxiv
        """
        results_path = Path(results_dir)
        summary_file = results_path / "summary.json"
        
        if not summary_file.exists():
            raise FileNotFoundError(f"No summary.json found in {results_dir}")
            
        with open(summary_file, 'r') as f:
            data = json.load(f)
        
        # Extract estimates (use bootstrap mean if available, otherwise point estimate)
        if 'priors' in data:
            if 'mean' in data['priors']:
                estimates_list = data['priors']['mean']
                ci_lo = data['priors'].get('ci_lo')
                ci_hi = data['priors'].get('ci_hi')
            else:
                estimates_list = data['priors']['point']
                ci_lo = ci_hi = None
        else:
            raise ValueError(f"No prior estimates found in {summary_file}")
        
        categories = data.get('categories', [])
        estimates = dict(zip(categories, estimates_list))
        
        # Handle confidence intervals
        confidence_intervals = None
        if ci_lo is not None and ci_hi is not None:
            confidence_intervals = {
                cat: (lo, hi) for cat, lo, hi in zip(categories, ci_lo, ci_hi)
            }
        
        # Map categories to ground truth format if needed
        estimates = self._map_categories(estimates)
        if confidence_intervals:
            confidence_intervals = self._map_categories_ci(confidence_intervals)
        
        method_name = results_path.name
        return MethodResult(
            name=method_name,
            estimates=estimates,
            confidence_intervals=confidence_intervals,
            metadata=data.get('config', {})
        )
    
    def _map_categories(self, estimates: Dict[str, float]) -> Dict[str, float]:
        """Map method categories to ground truth categories"""
        # Handle common mappings
        mapped = {}
        
        for gt_cat in self.ground_truth:
            if gt_cat in estimates:
                # Direct match - use as is
                mapped[gt_cat] = estimates[gt_cat]
            elif gt_cat == "GitHub" and "Code" in estimates:
                # Map Code -> GitHub
                mapped[gt_cat] = estimates["Code"]
            elif gt_cat == "Arxiv" and "Papers" in estimates:
                # Map Papers -> Arxiv  
                mapped[gt_cat] = estimates["Papers"]
            else:
                # Try case-insensitive matching
                for est_cat, val in estimates.items():
                    if est_cat.lower() == gt_cat.lower():
                        mapped[gt_cat] = val
                        break
                else:
                    # Category not found - warn user
                    print(f"Warning: Category '{gt_cat}' not found in method results. Setting to 0.0")
                    mapped[gt_cat] = 0.0
        
        return mapped
    
    def _map_categories_ci(self, cis: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
        """Map confidence intervals using same logic as estimates"""
        mapped = {}
        
        for gt_cat in self.ground_truth:
            if gt_cat in cis:
                # Direct match - use as is
                mapped[gt_cat] = cis[gt_cat]
            elif gt_cat == "GitHub" and "Code" in cis:
                # Map Code -> GitHub
                mapped[gt_cat] = cis["Code"]
            elif gt_cat == "Arxiv" and "Papers" in cis:
                # Map Papers -> Arxiv
                mapped[gt_cat] = cis["Papers"]
            else:
                # Try case-insensitive matching
                for est_cat, val in cis.items():
                    if est_cat.lower() == gt_cat.lower():
                        mapped[gt_cat] = val
                        break
                else:
                    # Category not found
                    mapped[gt_cat] = (0.0, 0.0)
        
        return mapped
    
    def evaluate_method(self, results_dir: str) -> Tuple[MethodResult, BenchmarkMetrics]:
        """Evaluate a single method against ground truth"""
        method_result = self._load_method_results(results_dir)
        metrics = self._compute_metrics(method_result)
        return method_result, metrics
    
    def _compute_metrics(self, method_result: MethodResult) -> BenchmarkMetrics:
        """Compute evaluation metrics for a method"""
        # Align categories and get arrays
        true_probs = np.array([self.ground_truth[cat] for cat in self.categories])
        est_probs = np.array([method_result.estimates.get(cat, 0.0) for cat in self.categories])
        
        # Basic distance metrics
        l1_distance = np.sum(np.abs(true_probs - est_probs))
        l2_distance = np.sqrt(np.sum((true_probs - est_probs) ** 2))
        
        # KL divergence (add small epsilon to avoid log(0))
        epsilon = 1e-10
        kl_div = np.sum(true_probs * np.log((true_probs + epsilon) / (est_probs + epsilon)))
        
        # Error metrics
        abs_errors = np.abs(true_probs - est_probs)
        max_abs_error = np.max(abs_errors)
        mean_abs_error = np.mean(abs_errors)
        
        # Relative errors (avoid division by zero)
        rel_errors = {}
        for i, cat in enumerate(self.categories):
            if true_probs[i] > 0:
                rel_errors[cat] = abs_errors[i] / true_probs[i]
            else:
                rel_errors[cat] = float('inf') if abs_errors[i] > 0 else 0.0
        
        # Confidence interval metrics (if available)
        coverage_rate = None
        interval_widths = None
        if method_result.confidence_intervals:
            coverage_count = 0
            widths = {}
            for i, cat in enumerate(self.categories):
                if cat in method_result.confidence_intervals:
                    ci_lo, ci_hi = method_result.confidence_intervals[cat]
                    if ci_lo <= true_probs[i] <= ci_hi:
                        coverage_count += 1
                    widths[cat] = ci_hi - ci_lo
                else:
                    widths[cat] = 0.0
            coverage_rate = coverage_count / len(self.categories)
            interval_widths = widths
        
        return BenchmarkMetrics(
            l1_distance=l1_distance,
            l2_distance=l2_distance,
            kl_divergence=kl_div,
            max_absolute_error=max_abs_error,
            mean_absolute_error=mean_abs_error,
            relative_errors=rel_errors,
            coverage_rate=coverage_rate,
            interval_widths=interval_widths
        )
    
    def compare_methods(self, results_dirs: List[str]) -> pd.DataFrame:
        """Compare multiple methods and return summary table"""
        results = []
        
        for results_dir in results_dirs:
            try:
                method_result, metrics = self.evaluate_method(results_dir)
                
                row = {
                    'Method': method_result.name,
                    'L1 Distance': f"{metrics.l1_distance:.4f}",
                    'L2 Distance': f"{metrics.l2_distance:.4f}",
                    'KL Divergence': f"{metrics.kl_divergence:.4f}",
                    'Max Abs Error': f"{metrics.max_absolute_error:.4f}",
                    'Mean Abs Error': f"{metrics.mean_absolute_error:.4f}",
                }
                
                if metrics.coverage_rate is not None:
                    row['CI Coverage'] = f"{metrics.coverage_rate:.2%}"
                    avg_width = np.mean(list(metrics.interval_widths.values()))
                    row['Avg CI Width'] = f"{avg_width:.4f}"
                
                results.append(row)
                
            except Exception as e:
                print(f"Warning: Failed to evaluate {results_dir}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def plot_comparison(self, results_dirs: List[str], output_path: str = "benchmark_comparison.png"):
        """Create comprehensive comparison plots"""
        methods_data = []
        
        for results_dir in results_dirs:
            try:
                method_result, metrics = self.evaluate_method(results_dir)
                methods_data.append((method_result, metrics))
            except Exception as e:
                print(f"Warning: Failed to load {results_dir}: {e}")
                continue
        
        if not methods_data:
            print("No valid methods to plot")
            return
        
        # Create subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Label-Shift Estimation Methods Comparison', fontsize=16)
        
        # Plot 1: Estimates vs Ground Truth
        ax1 = axes[0, 0]
        x = np.arange(len(self.categories))
        width = 0.8 / (len(methods_data) + 1)
        
        # Ground truth bars
        true_probs = [self.ground_truth[cat] for cat in self.categories]
        ax1.bar(x - width * len(methods_data)/2, true_probs, width, 
                label='Ground Truth', color='black', alpha=0.7)
        
        # Method estimates
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods_data)))
        for i, (method_result, _) in enumerate(methods_data):
            est_probs = [method_result.estimates.get(cat, 0.0) for cat in self.categories]
            ax1.bar(x - width * len(methods_data)/2 + width * (i+1), est_probs, width,
                   label=method_result.name, color=colors[i], alpha=0.8)
            
            # Add confidence intervals if available
            if method_result.confidence_intervals:
                cis = [method_result.confidence_intervals.get(cat, (0, 0)) for cat in self.categories]
                yerr = [(est - ci[0], ci[1] - est) for est, ci in zip(est_probs, cis)]
                yerr = np.array(yerr).T
                ax1.errorbar(x - width * len(methods_data)/2 + width * (i+1), est_probs, 
                           yerr=yerr, fmt='none', color='black', capsize=3, alpha=0.6)
        
        ax1.set_xlabel('Categories')
        ax1.set_ylabel('Proportion')
        ax1.set_title('Estimated vs Ground Truth Proportions')
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.categories, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Error by Category
        ax2 = axes[0, 1]
        for i, (method_result, metrics) in enumerate(methods_data):
            errors = [abs(method_result.estimates.get(cat, 0.0) - self.ground_truth[cat]) 
                     for cat in self.categories]
            ax2.plot(range(len(self.categories)), errors, 'o-', 
                    label=method_result.name, color=colors[i])
        
        ax2.set_xlabel('Categories')
        ax2.set_ylabel('Absolute Error')
        ax2.set_title('Absolute Error by Category')
        ax2.set_xticks(range(len(self.categories)))
        ax2.set_xticklabels(self.categories, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Overall Metrics Comparison
        ax3 = axes[1, 0]
        metrics_names = ['L1 Distance', 'L2 Distance', 'Mean Abs Error', 'Max Abs Error']
        metrics_data = []
        method_names = []
        
        for method_result, metrics in methods_data:
            method_names.append(method_result.name)
            metrics_data.append([
                metrics.l1_distance,
                metrics.l2_distance, 
                metrics.mean_absolute_error,
                metrics.max_absolute_error
            ])
        
        metrics_df = pd.DataFrame(metrics_data, columns=metrics_names, index=method_names)
        sns.heatmap(metrics_df, annot=True, fmt='.4f', cmap='Reds', ax=ax3)
        ax3.set_title('Error Metrics Heatmap')
        
        # Plot 4: Relative Errors
        ax4 = axes[1, 1]
        for i, (method_result, metrics) in enumerate(methods_data):
            rel_errors = [metrics.relative_errors.get(cat, 0) for cat in self.categories]
            # Cap relative errors for visualization
            rel_errors = [min(err, 5.0) if err != float('inf') else 5.0 for err in rel_errors]
            ax4.bar(x + width * i, rel_errors, width, 
                   label=method_result.name, color=colors[i], alpha=0.8)
        
        ax4.set_xlabel('Categories')
        ax4.set_ylabel('Relative Error (capped at 5.0)')
        ax4.set_title('Relative Error by Category')
        ax4.set_xticks(x + width * (len(methods_data)-1) / 2)
        ax4.set_xticklabels(self.categories, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Comparison plot saved to {output_path}")
    
    def generate_report(self, results_dirs: List[str], output_file: str = "benchmark_report.md"):
        """Generate a comprehensive markdown report"""
        report_lines = [
            "# Label-Shift Estimation Benchmark Report",
            "",
            f"**Ground Truth Model**: {Path(self.ground_truth_file).stem}",
            f"**Categories**: {', '.join(self.categories)}",
            "",
            "## Ground Truth Proportions",
            "",
        ]
        
        # Ground truth table
        report_lines.append("| Category | Proportion |")
        report_lines.append("|----------|------------|")
        for cat in self.categories:
            report_lines.append(f"| {cat} | {self.ground_truth[cat]:.4f} |")
        report_lines.append("")
        
        # Method comparison table
        comparison_df = self.compare_methods(results_dirs)
        if not comparison_df.empty:
            report_lines.append("## Method Comparison")
            report_lines.append("")
            report_lines.append(comparison_df.to_markdown(index=False))
            report_lines.append("")
        
        # Detailed results for each method
        report_lines.append("## Detailed Results")
        report_lines.append("")
        
        for results_dir in results_dirs:
            try:
                method_result, metrics = self.evaluate_method(results_dir)
                
                report_lines.extend([
                    f"### {method_result.name}",
                    "",
                    "**Estimates vs Ground Truth:**",
                    "",
                    "| Category | Estimate | Ground Truth | Absolute Error | Relative Error |",
                    "|----------|----------|--------------|----------------|----------------|"
                ])
                
                for cat in self.categories:
                    est = method_result.estimates.get(cat, 0.0)
                    true_val = self.ground_truth[cat]
                    abs_err = abs(est - true_val)
                    rel_err = metrics.relative_errors[cat]
                    rel_err_str = f"{rel_err:.2%}" if rel_err != float('inf') else "∞"
                    
                    ci_str = ""
                    if method_result.confidence_intervals and cat in method_result.confidence_intervals:
                        ci_lo, ci_hi = method_result.confidence_intervals[cat]
                        ci_str = f" [{ci_lo:.4f}, {ci_hi:.4f}]"
                    
                    report_lines.append(
                        f"| {cat} | {est:.4f}{ci_str} | {true_val:.4f} | {abs_err:.4f} | {rel_err_str} |"
                    )
                
                report_lines.extend([
                    "",
                    "**Summary Metrics:**",
                    f"- L1 Distance: {metrics.l1_distance:.4f}",
                    f"- L2 Distance: {metrics.l2_distance:.4f}",
                    f"- KL Divergence: {metrics.kl_divergence:.4f}",
                    f"- Mean Absolute Error: {metrics.mean_absolute_error:.4f}",
                    f"- Max Absolute Error: {metrics.max_absolute_error:.4f}",
                ])
                
                if metrics.coverage_rate is not None:
                    report_lines.append(f"- Confidence Interval Coverage: {metrics.coverage_rate:.2%}")
                    avg_width = np.mean(list(metrics.interval_widths.values()))
                    report_lines.append(f"- Average CI Width: {avg_width:.4f}")
                
                report_lines.append("")
                
            except Exception as e:
                report_lines.extend([
                    f"### {Path(results_dir).name} (Failed)",
                    f"Error: {e}",
                    ""
                ])
        
        # Write report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Benchmark report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark evaluation for label-shift estimation methods")
    parser.add_argument("--results_dir", type=str, help="Single method results directory to evaluate")
    parser.add_argument("--compare", nargs='+', help="Multiple method directories to compare")
    parser.add_argument("--ground_truth", type=str, required=True, help="Ground truth YAML file")
    parser.add_argument("--output_dir", type=str, default="benchmark_output", help="Output directory for plots and reports")
    parser.add_argument("--no_plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--no_report", action="store_true", help="Skip generating markdown report")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize benchmark
    benchmark = LabelShiftBenchmark(args.ground_truth)
    
    # Determine methods to evaluate
    if args.results_dir:
        methods = [args.results_dir]
    elif args.compare:
        methods = args.compare
    else:
        parser.error("Must specify either --results_dir or --compare")
    
    # Run evaluation
    if len(methods) == 1:
        # Single method evaluation
        method_result, metrics = benchmark.evaluate_method(methods[0])
        print(f"\n=== Evaluation Results for {method_result.name} ===")
        print(f"L1 Distance: {metrics.l1_distance:.4f}")
        print(f"L2 Distance: {metrics.l2_distance:.4f}")
        print(f"KL Divergence: {metrics.kl_divergence:.4f}")
        print(f"Mean Absolute Error: {metrics.mean_absolute_error:.4f}")
        print(f"Max Absolute Error: {metrics.max_absolute_error:.4f}")
        
        if metrics.coverage_rate is not None:
            print(f"CI Coverage Rate: {metrics.coverage_rate:.2%}")
            avg_width = np.mean(list(metrics.interval_widths.values()))
            print(f"Average CI Width: {avg_width:.4f}")
    else:
        # Multiple method comparison
        comparison_df = benchmark.compare_methods(methods)
        print("\n=== Method Comparison ===")
        print(comparison_df.to_string(index=False))
    
    # Generate plots
    if not args.no_plots:
        plot_path = output_dir / "comparison.png"
        benchmark.plot_comparison(methods, str(plot_path))
    
    # Generate report
    if not args.no_report:
        report_path = output_dir / "report.md"
        benchmark.generate_report(methods, str(report_path))


if __name__ == "__main__":
    main()
