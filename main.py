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

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    # Data loading parameters
    VAL_RATIO = 0.2                 # 20% of training data for validation
    MISSINGNESS_THRESHOLD = 0.6    # Drop features with >60% missing
    SEED = 42

    # Preprocessing
    REMOVE_OUTLIERS = True         # Remove outliers using IQR method
    NORMALIZE = True               # Normalize using StandardScaler
    SAVE_DATA_XLSX = True          # Save train_val and test splits as XLSX

    # Experiment parameters
    MASK_RATIO = 0.2               # Mask 20% of observed values

    # Select masking strategy (choose one)
    MASKING_STRATEGY = MaskingStrategy.MCAR
    # MASKING_STRATEGY = MaskingStrategy.SEQUENCE_END
    # MASKING_STRATEGY = MaskingStrategy.VARIABLE_WISE

    # Select imputation method (choose one)
    IMPUTATION_METHOD = ImputationMethod.SMOOTHING_SPLINE
    # IMPUTATION_METHOD = ImputationMethod.GAUSSIAN_PROCESS
    # IMPUTATION_METHOD = ImputationMethod.MICE

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
    # RUN SINGLE EXPERIMENT
    # =========================================================================

    config = ExperimentConfig(
        masking_strategy=MASKING_STRATEGY,
        imputation_method=IMPUTATION_METHOD,
        mask_ratio=MASK_RATIO,
        seed=SEED
    )

    # Evaluate on validation set
    result = run_experiment(
        data=data,
        config=config,
        evaluate_on=['val'],
        save_results=True,
        output_dir=OUTPUT_DIR,
        verbose=True
    )

    # Print detailed results
    print("\n" + summarize_results(result.val_metrics))

    print(f"\n{'='*60}")
    print("Output Files Generated:")
    print("="*60)
    print(f"  Directory: {OUTPUT_DIR}")
    print(f"  - train_val_data.xlsx: Preprocessed training+validation data")
    print(f"  - test_data.xlsx: Preprocessed test data")
    print(f"  - *_overall.xlsx: Overall metrics")
    print(f"  - *_per_variable.xlsx: Per-variable metrics")
    print(f"  - *_imputed_data.xlsx: Imputed dataset")
    print(f"  - *_overall_scatter.png: Predicted vs actual plot")
    print(f"  - *_per_variable_scatter.png: Per-variable scatter plots")


def run_full_comparison():
    """
    Run a full comparison across all masking strategies and imputation methods.
    """
    DATA_DIR = Path(__file__).parent / "data"
    OUTPUT_DIR = Path(__file__).parent / "output"

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
            ImputationMethod.SMOOTHING_SPLINE,
            ImputationMethod.GAUSSIAN_PROCESS,
            ImputationMethod.MICE
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
    # Run single experiment with configured parameters
    main()

    # Uncomment below to run full comparison:
    # run_full_comparison()
