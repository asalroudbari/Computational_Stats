"""
Irregular Multivariate Time Series Missing Value Imputation
PhysioNet 2012 Challenge Dataset - ICU Patient Data

This project compares three imputation methods:
1. Smoothing Splines
2. Gaussian Processes
3. Bayesian Imputation (MICE)

Under three masking strategies:
1. Missing Completely At Random (MCAR) - dataset-wide
2. Sequence-end masking - patient-wise
3. Variable-wise masking - dataset-wide
"""

from pathlib import Path
import os  # Add this import to handle directory creation

from dataloader import load_physionet_data, print_data_summary
from masking import MaskingStrategy
from imputation import ImputationMethod
from experiment import (
    ExperimentConfig,
    run_experiment,
    run_experiment_grid,
    print_results_table,
    summarize_results
)


def main():
    DATA_DIR = Path(__file__).parent / "data"
    OUTPUT_DIR = Path(__file__).parent / "output"

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    # Data loading parameters
    VAL_RATIO = 0.2                 # 20% of training data for validation
    MISSINGNESS_THRESHOLD = 0.7    # Drop features with >60% missing
    SEED = 42

    # Preprocessing
    REMOVE_OUTLIERS = True         # Remove outliers using IQR method
    NORMALIZE = True               # Normalize using StandardScaler
    SAVE_DATA_XLSX = True          # Save train_val and test splits as XLSX

    # Experiment parameters
    MASK_RATIO = 0.2               # Mask 20% of observed values

    # Select masking strategy (choose one)
    MASKING_STRATEGY = MaskingStrategy.MCAR

    # List of imputation methods to evaluate
    IMPUTATION_METHODS = [
        ImputationMethod.SMOOTHING_SPLINE,
        ImputationMethod.GAUSSIAN_PROCESS,
        ImputationMethod.MICE,
        ImputationMethod.HMM
    ]

    # =========================================================================
    # LOAD DATA
    # =========================================================================

    print("=" * 60)
    print("IRREGULAR TIME SERIES IMPUTATION - PhysioNet 2012")
    print("=" * 60)

    data = load_physionet_data(
        data_dir=DATA_DIR,
        val_ratio=VAL_RATIO,
        missingness_threshold=MISSINGNESS_THRESHOLD,
        seed=SEED,
        remove_outliers=REMOVE_OUTLIERS,
        normalize=NORMALIZE,
        save_xlsx=SAVE_DATA_XLSX,
        output_dir=OUTPUT_DIR,
        verbose=True
    )

    print_data_summary(data)

    # =========================================================================
    # RUN EXPERIMENTS FOR ALL IMPUTATION METHODS
    # =========================================================================

    all_results = []

    for method in IMPUTATION_METHODS:
        print(f"\n{'='*60}")
        print(f"Running experiment with {method.name}")
        print(f"{'='*60}")

        config = ExperimentConfig(
            masking_strategy=MASKING_STRATEGY,
            imputation_method=method,
            mask_ratio=MASK_RATIO,
            seed=SEED
        )

        # Ensure method-specific output directory exists
        method_output_dir = OUTPUT_DIR / method.name
        os.makedirs(method_output_dir, exist_ok=True)

        # Evaluate on validation set
        try:
            result = run_experiment(
                data=data,
                config=config,
                evaluate_on=['val'],
                save_results=True,
                output_dir=method_output_dir,
                verbose=True
            )
        except ImportError as exc:
            print(f"Skipping method '{method.value}': {exc}")
            continue

        # Store the full result object for downstream processing
        all_results.append(result)

        # Print detailed results
        print("\n" + summarize_results(result.val_metrics))

    # =========================================================================
    # SAVE COMPARISON METRICS
    # =========================================================================

    from experiment import results_to_dataframe
    summary_df = results_to_dataframe(all_results, 'val')
    summary_df.to_excel(OUTPUT_DIR / "imputation_comparison_summary.xlsx", index=False)
    print(f"\nComparison summary saved to {OUTPUT_DIR / 'imputation_comparison_summary.xlsx'}")


def run_full_comparison():
    """
    Run a full comparison across all masking strategies and imputation methods.
    """
    DATA_DIR = Path(__file__).parent / "data"
    OUTPUT_DIR = Path(__file__).parent / "output"

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("FULL COMPARISON - All Strategies & Methods")
    print("=" * 60)

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

    print_data_summary(data)

    # Run grid of experiments
    results = run_experiment_grid(
        data=data,
        masking_strategies=[
            MaskingStrategy.MCAR,
            MaskingStrategy.SEQUENCE_END,
            MaskingStrategy.VARIABLE_WISE
        ],
        imputation_methods=[
            # ImputationMethod.SMOOTHING_SPLINE,
            # ImputationMethod.GAUSSIAN_PROCESS,
            # ImputationMethod.MICE,
            ImputationMethod.HMM
        ],
        mask_ratios=[0.2],
        evaluate_on=['val'],
        save_results=True,
        output_dir=OUTPUT_DIR,
        seed=42,
        verbose=True
    )

    # Print summary table
    print_results_table(results, split='val')

    # Save summary to Excel
    from experiment import results_to_dataframe
    summary_df = results_to_dataframe(results, 'val')
    summary_df.to_excel(OUTPUT_DIR / "experiment_summary.xlsx", index=False)
    print(f"\nSummary saved to {OUTPUT_DIR / 'experiment_summary.xlsx'}")

    return results


if __name__ == "__main__":
    # Run experiments for all imputation methods
    main()

    # Uncomment below to run full comparison:
    # run_full_comparison()