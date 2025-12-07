"""
Experiment Pipeline for Imputation Evaluation

Orchestrates the full pipeline: masking -> imputation -> evaluation
with results saving and visualization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from dataloader import PhysioNetData, DataSplit, pivot_timeseries
from masking import MaskingStrategy, MaskedData, mask_dataset
from imputation import ImputationMethod, impute_patient_timeseries
from evaluation import PatientMetrics, evaluate_patient_imputation, summarize_results


# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    masking_strategy: MaskingStrategy
    imputation_method: ImputationMethod
    mask_ratio: float = 0.2
    seed: int = 42
    imputation_params: Dict = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    config: ExperimentConfig
    train_metrics: PatientMetrics = None
    val_metrics: Optional[PatientMetrics] = None
    test_metrics: Optional[PatientMetrics] = None
    runtime_seconds: float = 0.0
    # Store actual vs predicted for plotting
    actual_values: np.ndarray = None
    predicted_values: np.ndarray = None
    per_variable_results: Dict[str, Dict] = field(default_factory=dict)


def collect_actual_predicted(masked_data: Dict[int, MaskedData],
                              imputed_data: Dict[int, pd.DataFrame],
                              temporal_features: List[str] = None) -> tuple:
    """Collect all actual and predicted values from masked positions.

    Args:
        masked_data: Dict mapping patient_id to MaskedData
        imputed_data: Dict mapping patient_id to imputed DataFrame
        temporal_features: List of temporal features to include (if None, all)

    Returns:
        Tuple of (all_actual, all_predicted, per_variable)
    """
    all_actual = []
    all_predicted = []
    per_variable = {}

    for pid in masked_data.keys():
        if pid not in imputed_data:
            continue

        mdata = masked_data[pid]
        imputed = imputed_data[pid]

        if not hasattr(mdata, 'original'):
            continue

        original = mdata.original
        mask = mdata.mask
        
        # First pass: collect all patient data
        patient_actual = []
        patient_predicted = []

        for col in original.columns:
            # Skip if not a temporal feature (when specified)
            if temporal_features is not None and col not in temporal_features:
                continue
            if col not in mask.columns:
                continue

            col_mask = mask[col]
            if col_mask.sum() == 0:
                continue

            actual = original[col].values[col_mask.values]
            predicted = imputed[col].values[col_mask.values]

            # Filter NaN
            valid = ~(np.isnan(actual) | np.isnan(predicted))
            patient_actual.extend(actual[valid])
            patient_predicted.extend(predicted[valid])

        # Collect all patient data without filtering
        if len(patient_actual) > 0:
            all_actual.extend(patient_actual)
            all_predicted.extend(patient_predicted)
        
        # Second pass: collect per-variable with same filter applied per variable
        for col in original.columns:
            if temporal_features is not None and col not in temporal_features:
                continue
            if col not in mask.columns:
                continue

            col_mask = mask[col]
            if col_mask.sum() == 0:
                continue

            actual = original[col].values[col_mask.values]
            predicted = imputed[col].values[col_mask.values]

            valid = ~(np.isnan(actual) | np.isnan(predicted))
            actual_valid = actual[valid]
            predicted_valid = predicted[valid]

            # Collect per-variable data without filtering
            if len(actual_valid) > 0:
                if col not in per_variable:
                    per_variable[col] = {'actual': [], 'predicted': []}
                per_variable[col]['actual'].extend(actual_valid)
                per_variable[col]['predicted'].extend(predicted_valid)

    return np.array(all_actual), np.array(all_predicted), per_variable


def plot_overall_scatter(actual: np.ndarray, predicted: np.ndarray,
                         config: ExperimentConfig, output_path: Path) -> None:
    """Create overall predicted vs actual scatter plot."""
    # Outlier filter already applied in collect_actual_predicted
    fig, ax = plt.subplots(figsize=(8, 8))

    # Use hexbin for better visualization of dense regions
    # Calculate point density for coloring
    from scipy.stats import gaussian_kde
    if len(actual) > 2:
        try:
            xy = np.vstack([actual, predicted])
            z = gaussian_kde(xy)(xy)
            # Sort points by density so dense points are plotted last
            idx = z.argsort()
            actual_sorted, predicted_sorted, z_sorted = actual[idx], predicted[idx], z[idx]
            scatter = ax.scatter(actual_sorted, predicted_sorted, c=z_sorted, s=10, 
                                alpha=0.5, cmap='viridis', edgecolors='none')
            plt.colorbar(scatter, ax=ax, label='Point Density')
        except:
            # Fallback to regular scatter if KDE fails
            ax.scatter(actual, predicted, alpha=0.3, s=10, c='steelblue')
    else:
        ax.scatter(actual, predicted, alpha=0.3, s=10, c='steelblue')

    # Perfect prediction line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction', zorder=10)

    # Set axis limits to focus on central 99% of data (remove extreme outliers from view)
    actual_p1, actual_p99 = np.percentile(actual, [1, 99])
    predicted_p1, predicted_p99 = np.percentile(predicted, [1, 99])
    axis_min = min(actual_p1, predicted_p1)
    axis_max = max(actual_p99, predicted_p99)
    margin = (axis_max - axis_min) * 0.1
    ax.set_xlim(axis_min - margin, axis_max + margin)
    ax.set_ylim(axis_min - margin, axis_max + margin)

    # Compute metrics for annotation
    mae = np.mean(np.abs(predicted - actual))
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    ss_res = np.sum((predicted - actual) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    # Add text box with metrics
    textstr = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}\nN: {len(actual)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    ax.set_xlabel('Actual Values (Normalized)', fontsize=12)
    ax.set_ylabel('Predicted Values (Normalized)', fontsize=12)
    ax.set_title(f'Overall: {config.imputation_method.value} with {config.masking_strategy.value} masking',
                 fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_variable_scatter(per_variable: Dict[str, Dict],
                               config: ExperimentConfig,
                               output_path: Path) -> None:
    """Create per-variable predicted vs actual scatter plots."""
    n_vars = len(per_variable)
    if n_vars == 0:
        return

    # Calculate grid size
    n_cols = min(3, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_vars == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (var, data) in enumerate(sorted(per_variable.items())):
        ax = axes[i]
        actual = np.array(data['actual'])
        predicted = np.array(data['predicted'])

        ax.scatter(actual, predicted, alpha=0.4, s=15, c='steelblue')

        # Perfect prediction line
        if len(actual) > 0:
            min_val = min(actual.min(), predicted.min())
            max_val = max(actual.max(), predicted.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

            # Compute metrics
            mae = np.mean(np.abs(predicted - actual))
            rmse = np.sqrt(np.mean((predicted - actual) ** 2))
            ss_res = np.sum((predicted - actual) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

            textstr = f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r2:.3f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props)

        ax.set_xlabel('Actual', fontsize=10)
        ax.set_ylabel('Predicted', fontsize=10)
        ax.set_title(f'{var} (n={len(actual)})', fontsize=11)

    # Hide unused subplots
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'Per-Variable: {config.imputation_method.value} with {config.masking_strategy.value}',
                 fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_results_xlsx(result: ExperimentResult, metrics: PatientMetrics,
                       output_dir: Path, split_name: str) -> None:
    """Save experiment results to Excel files."""
    prefix = f"{result.config.masking_strategy.value}_{result.config.imputation_method.value}"

    # Save overall metrics
    overall_df = pd.DataFrame([{
        'masking_strategy': result.config.masking_strategy.value,
        'imputation_method': result.config.imputation_method.value,
        'mask_ratio': result.config.mask_ratio,
        'mae': metrics.overall.mae,
        'mse': metrics.overall.mse,
        'rmse': metrics.overall.rmse,
        'r2': metrics.overall.r2,
        'n_evaluated': metrics.overall.n_evaluated,
        'runtime_seconds': result.runtime_seconds
    }])
    overall_df.to_excel(output_dir / f"{prefix}_{split_name}_overall.xlsx", index=False)

    # Save per-variable metrics
    if metrics.per_variable:
        var_rows = []
        for var, m in metrics.per_variable.items():
            var_rows.append({
                'variable': var,
                'mae': m.mae,
                'mse': m.mse,
                'rmse': m.rmse,
                'r2': m.r2,
                'n_evaluated': m.n_evaluated
            })
        var_df = pd.DataFrame(var_rows)
        var_df.to_excel(output_dir / f"{prefix}_{split_name}_per_variable.xlsx", index=False)


def save_imputed_data_xlsx(masked_data: Dict[int, MaskedData],
                            imputed_data: Dict[int, pd.DataFrame],
                            features: List[str],
                            output_path: Path) -> None:
    """Save imputed dataset to Excel file."""
    rows = []

    for pid in masked_data.keys():
        if pid not in imputed_data:
            continue

        mdata = masked_data[pid]
        imputed = imputed_data[pid]

        for t in imputed.index:
            row = {'patient_id': pid, 'time_hours': t}
            for f in features:
                if f in imputed.columns:
                    row[f] = imputed.loc[t, f]
                else:
                    row[f] = np.nan
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(['patient_id', 'time_hours']).reset_index(drop=True)
    df.to_excel(output_path, index=False)


def run_experiment(data: PhysioNetData,
                   config: ExperimentConfig,
                   evaluate_on: List[str] = ['val'],
                   save_results: bool = True,
                   output_dir: Optional[Path] = None,
                   verbose: bool = True) -> ExperimentResult:
    """
    Run a single experiment with specified masking and imputation.

    Args:
        data: PhysioNetData containing train/val/test splits
        config: Experiment configuration
        evaluate_on: Which splits to evaluate ('train', 'val', 'test')
        save_results: Whether to save results to files
        output_dir: Directory for output files
        verbose: Print progress

    Returns:
        ExperimentResult with metrics for each evaluated split
    """
    start_time = time.time()

    if output_dir is None:
        output_dir = Path("output")
    output_dir = Path(output_dir)

    if save_results:
        output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Running Experiment")
        print(f"  Masking:    {config.masking_strategy.value}")
        print(f"  Imputation: {config.imputation_method.value}")
        print(f"  Mask Ratio: {config.mask_ratio:.0%}")
        print(f"{'='*60}")

    result = ExperimentResult(config=config)

    # Process each split
    splits_to_process = []
    if 'train' in evaluate_on:
        splits_to_process.append(('train', data.train))
    if 'val' in evaluate_on:
        splits_to_process.append(('val', data.val))
    if 'test' in evaluate_on:
        splits_to_process.append(('test', data.test))

    for split_name, split_data in splits_to_process:
        if verbose:
            print(f"\nProcessing {split_name} split ({split_data.n_patients} patients)...")

        # Step 1: Apply masking (dataset-wide for MCAR/VARIABLE_WISE)
        if verbose:
            print(f"  Applying {config.masking_strategy.value} masking...")

        # Only mask temporal features (not static features)
        # Static features (Age, Gender, Height, ICUType) added but never masked
        masked_data = mask_dataset(
            timeseries=split_data.timeseries,
            strategy=config.masking_strategy,
            features=data.temporal_features,  # Only mask temporal features
            mask_ratio=config.mask_ratio,
            seed=config.seed,
            pivot_fn=pivot_timeseries,
            general_info=split_data.general_info  # Add static features
        )

        total_masked = sum(m.n_masked for m in masked_data.values())
        if verbose:
            print(f"    Masked {total_masked} values across all patients")

        # Step 2: Apply imputation
        if verbose:
            print(f"  Applying {config.imputation_method.value} imputation...")

        imputed_data = impute_patient_timeseries(
            masked_data=masked_data,
            method=config.imputation_method,
            **config.imputation_params
        )

        # Step 3: Evaluate
        if verbose:
            print(f"  Evaluating imputation quality...")

        metrics = evaluate_patient_imputation(masked_data, imputed_data)

        if verbose:
            print(f"    Overall MAE:  {metrics.overall.mae:.4f}")
            print(f"    Overall RMSE: {metrics.overall.rmse:.4f}")
            print(f"    Overall R²:   {metrics.overall.r2:.4f}")

        # Collect actual vs predicted for plotting (temporal features only)
        actual, predicted, per_var = collect_actual_predicted(
            masked_data, imputed_data, data.temporal_features
        )
        result.actual_values = actual
        result.predicted_values = predicted
        result.per_variable_results = per_var

        # Store metrics
        if split_name == 'train':
            result.train_metrics = metrics
        elif split_name == 'val':
            result.val_metrics = metrics
        elif split_name == 'test':
            result.test_metrics = metrics

        # Save results
        if save_results:
            prefix = f"{config.masking_strategy.value}_{config.imputation_method.value}"

            # Save plots
            if len(actual) > 0:
                plot_overall_scatter(actual, predicted, config,
                                     output_dir / f"{prefix}_{split_name}_overall_scatter.png")
                plot_per_variable_scatter(per_var, config,
                                          output_dir / f"{prefix}_{split_name}_per_variable_scatter.png")

            # Save metrics to Excel
            save_results_xlsx(result, metrics, output_dir, split_name)

            # Save imputed data (temporal features only)
            save_imputed_data_xlsx(masked_data, imputed_data, data.temporal_features,
                                   output_dir / f"{prefix}_{split_name}_imputed_data.xlsx")

            if verbose:
                print(f"    Saved results to {output_dir}")

    result.runtime_seconds = time.time() - start_time

    if verbose:
        print(f"\nExperiment completed in {result.runtime_seconds:.1f} seconds")

    return result


def run_experiment_grid(data: PhysioNetData,
                        masking_strategies: List[MaskingStrategy],
                        imputation_methods: List[ImputationMethod],
                        mask_ratios: List[float] = [0.2],
                        evaluate_on: List[str] = ['val'],
                        save_results: bool = True,
                        output_dir: Optional[Path] = None,
                        seed: int = 42,
                        verbose: bool = True) -> List[ExperimentResult]:
    """Run experiments over a grid of configurations."""
    results = []
    total_experiments = len(masking_strategies) * len(imputation_methods) * len(mask_ratios)
    exp_num = 0

    for mask_ratio in mask_ratios:
        for masking in masking_strategies:
            for imputation in imputation_methods:
                exp_num += 1
                if verbose:
                    print(f"\n[{exp_num}/{total_experiments}]", end="")

                config = ExperimentConfig(
                    masking_strategy=masking,
                    imputation_method=imputation,
                    mask_ratio=mask_ratio,
                    seed=seed
                )

                result = run_experiment(data, config, evaluate_on,
                                        save_results, output_dir, verbose)
                results.append(result)

    return results


def results_to_dataframe(all_results, split: str):
    # all_results may contain plain result objects or tuples like (result, config, ...)
    processed_results = []
    for item in all_results:
        # Unpack if tuple; assume first element is the actual result object
        r = item[0] if isinstance(item, tuple) and item else item
        processed_results.append(r)

    # Replace occurrences of "for r in all_results" with "for r in processed_results"
    rows = []
    for r in processed_results:
        if split == "val":
            metrics = r.val_metrics
        elif split == "test":
            metrics = r.test_metrics
        else:
            metrics = r.train_metrics
        # ...existing code that builds `rows` from `metrics`...
        if metrics is None:
            continue

        rows.append({
            'masking_strategy': r.config.masking_strategy.value,
            'imputation_method': r.config.imputation_method.value,
            'mask_ratio': r.config.mask_ratio,
            'mae': metrics.overall.mae,
            'mse': metrics.overall.mse,
            'rmse': metrics.overall.rmse,
            'r2': metrics.overall.r2,
            'n_evaluated': metrics.overall.n_evaluated,
            'runtime_seconds': r.runtime_seconds
        })

    return pd.DataFrame(rows)


def print_results_table(results: List[ExperimentResult],
                        split: str = 'val') -> None:
    """Print a formatted table of results."""
    df = results_to_dataframe(results, split)

    if len(df) == 0:
        print("No results to display.")
        return

    print(f"\n{'='*80}")
    print(f"EXPERIMENT RESULTS SUMMARY ({split} split)")
    print('='*80)
    print(f"\n{'Masking':<18} {'Imputation':<18} {'Ratio':>6} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
    print("-" * 80)

    for _, row in df.iterrows():
        r2_str = f"{row['r2']:.4f}" if not np.isnan(row['r2']) else "N/A"
        print(f"{row['masking_strategy']:<18} {row['imputation_method']:<18} "
              f"{row['mask_ratio']:>6.0%} {row['mae']:>10.4f} {row['rmse']:>10.4f} {r2_str:>10}")


if __name__ == "__main__":
    from dataloader import load_physionet_data

    DATA_DIR = Path(__file__).parent / "data"
    OUTPUT_DIR = Path(__file__).parent / "output"

    print("Loading data...")
    data = load_physionet_data(
        data_dir=DATA_DIR,
        val_ratio=0.2,
        missingness_threshold=0.6,
        seed=42,
        remove_outliers=True,
        normalize=True,
        save_xlsx=True,
        output_dir=OUTPUT_DIR,
        verbose=True
    )

    # Run single experiment
    config = ExperimentConfig(
        masking_strategy=MaskingStrategy.MCAR,
        imputation_method=ImputationMethod.SMOOTHING_SPLINE,
        mask_ratio=0.2
    )

    result = run_experiment(
        data, config,
        evaluate_on=['val'],
        save_results=True,
        output_dir=OUTPUT_DIR,
        verbose=True
    )

    print("\n" + summarize_results(result.val_metrics))
