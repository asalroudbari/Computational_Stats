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
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from typing import Dict, List, Tuple

from dataloader import load_physionet_data, print_data_summary, load_raw_dataset, filter_features_by_missingness, filter_patient_timeseries, TEMPORAL_FEATURES, pivot_timeseries
from masking import MaskingStrategy, mask_dataset
from imputation import ImputationMethod, impute_patient_timeseries
from evaluation import evaluate_patient_imputation
from experiment import (
    ExperimentConfig,
    run_experiment,
    run_experiment_grid,
    print_results_table,
    summarize_results
)


def tune_hyperparameters(train_data: Dict[int, pd.DataFrame],
                         train_patient_ids: List[int],
                         temporal_features: List[str],
                         masking_strategy: MaskingStrategy,
                         imputation_method: ImputationMethod,
                         mask_ratio: float,
                         seed: int,
                         n_splits: int = 5) -> Dict:
    """
    Tune hyperparameters using GroupKFold cross-validation.

    Args:
        train_data: Dictionary mapping patient_id to timeseries DataFrame
        train_patient_ids: List of patient IDs for grouping
        temporal_features: List of temporal feature names to mask
        masking_strategy: Strategy for masking data
        imputation_method: Imputation method to tune
        mask_ratio: Ratio of values to mask
        seed: Random seed
        n_splits: Number of CV folds

    Returns:
        Dictionary with best hyperparameters
    """
    print(f"\nTuning hyperparameters for {imputation_method.name}...")

    # Define hyperparameter grids for each method
    param_grids = {
        ImputationMethod.SMOOTHING_SPLINE: {
            'smoothing_factor': [None, 0.1, 0.5, 1.0, 2.0]
        },
        ImputationMethod.GAUSSIAN_PROCESS: {
            'length_scale': [1.0, 5.0, 10.0, 20.0],
            'noise_level': [0.01, 0.1, 0.5]
        },
        ImputationMethod.MICE: {
            'max_iter': [5, 10, 20],
            'n_nearest_features': [None, 5, 10]
        },
        ImputationMethod.HMM: {
            'n_states': [2, 4, 6, 8],
            'max_iter': [50, 100, 200]
        }
    }

    param_grid = param_grids.get(imputation_method, {})

    if not param_grid:
        print(f"  No hyperparameters to tune for {imputation_method.name}")
        return {}

    # Generate all parameter combinations
    from itertools import product
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = [dict(zip(param_names, v)) for v in product(*param_values)]

    print(f"  Testing {len(param_combinations)} parameter combinations with {n_splits}-fold CV")

    # Setup GroupKFold
    gkf = GroupKFold(n_splits=n_splits)
    patient_ids_array = np.array(train_patient_ids)

    best_score = float('inf')
    best_params = {}
    results = []

    for params in param_combinations:
        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(patient_ids_array, groups=patient_ids_array)):
            fold_train_ids = patient_ids_array[train_idx].tolist()
            fold_val_ids = patient_ids_array[val_idx].tolist()

            fold_train_data = {pid: train_data[pid] for pid in fold_train_ids}
            fold_val_data = {pid: train_data[pid] for pid in fold_val_ids}

            # Mask validation data
            try:
                masked_val = mask_dataset(
                    fold_val_data,
                    strategy=masking_strategy,
                    features=temporal_features,
                    mask_ratio=mask_ratio,
                    seed=seed + fold_idx,
                    pivot_fn=pivot_timeseries
                )

                # Impute with current parameters
                imputed_val = impute_patient_timeseries(
                    masked_val,
                    method=imputation_method,
                    **params
                )

                # Evaluate
                metrics = evaluate_patient_imputation(masked_val, imputed_val)
                fold_scores.append(metrics.overall.rmse)

            except Exception as e:
                print(f"    Warning: Fold {fold_idx+1} failed with params {params}: {e}")
                fold_scores.append(float('inf'))

        avg_score = np.mean(fold_scores)
        results.append((params, avg_score))

        if avg_score < best_score:
            best_score = avg_score
            best_params = params

    print(f"  Best parameters: {best_params} (CV RMSE: {best_score:.4f})")

    return best_params


def load_combined_data(data_dir: Path,
                       missingness_threshold: float,
                       remove_outliers: bool,
                       normalize: bool,
                       test_ratio: float,
                       seed: int,
                       verbose: bool = True):
    """
    Load both set-a and set-b, combine them, and create train/test split.

    Returns:
        Tuple of (train_data, test_data, features_info)
    """
    if verbose:
        print("=" * 60)
        print("Loading Combined PhysioNet 2012 Dataset (set-a + set-b)")
        print("=" * 60)

    # Load both sets
    train_info_a, train_ts_a = load_raw_dataset(data_dir, 'set-a', verbose)
    train_info_b, train_ts_b = load_raw_dataset(data_dir, 'set-b', verbose)

    # Combine datasets
    combined_info = pd.concat([train_info_a, train_info_b], ignore_index=True)
    combined_ts = {**train_ts_a, **train_ts_b}

    if verbose:
        print(f"\nCombined dataset: {len(combined_ts)} patients")

    # Filter features by missingness on combined data
    kept_features, dropped_features = filter_features_by_missingness(
        combined_ts, missingness_threshold
    )
    temporal_features = [f for f in kept_features if f in TEMPORAL_FEATURES]

    if verbose:
        print(f"  Kept {len(kept_features)} features, dropped {len(dropped_features)} features")
        print(f"  Temporal features: {temporal_features}")

    # Filter time series
    combined_ts_filtered = filter_patient_timeseries(combined_ts, kept_features)

    # Split into train/test by patient
    np.random.seed(seed)
    all_patient_ids = list(combined_ts_filtered.keys())
    np.random.shuffle(all_patient_ids)

    n_test = int(len(all_patient_ids) * test_ratio)
    test_patient_ids = all_patient_ids[:n_test]
    train_patient_ids = all_patient_ids[n_test:]

    train_data = {pid: combined_ts_filtered[pid] for pid in train_patient_ids}
    test_data = {pid: combined_ts_filtered[pid] for pid in test_patient_ids}

    if verbose:
        print(f"\nTrain/Test Split:")
        print(f"  Train: {len(train_patient_ids)} patients ({len(train_patient_ids)/len(all_patient_ids)*100:.1f}%)")
        print(f"  Test:  {len(test_patient_ids)} patients ({len(test_patient_ids)/len(all_patient_ids)*100:.1f}%)")

    features_info = {
        'kept_features': kept_features,
        'dropped_features': dropped_features,
        'temporal_features': temporal_features
    }

    return train_data, train_patient_ids, test_data, test_patient_ids, features_info


def main():
    DATA_DIR = Path(__file__).parent / "data"
    OUTPUT_DIR = Path(__file__).parent / "output"

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    # Hyperparameter tuning flag
    TUNE = True  # If True, tune on combined data with GroupKFold and test on test set

    # Data loading parameters
    TEST_RATIO = 0.2                # 20% of data for test (when TUNE=True)
    VAL_RATIO = 0.2                 # 20% of training data for validation (when TUNE=False)
    MISSINGNESS_THRESHOLD = 0.7     # Drop features with >70% missing
    SEED = 42
    N_FOLDS = 5                     # Number of folds for GroupKFold CV

    # Preprocessing
    REMOVE_OUTLIERS = True          # Remove outliers using IQR method
    NORMALIZE = True                # Normalize using StandardScaler
    SAVE_DATA_XLSX = True           # Save train_val and test splits as XLSX

    # Experiment parameters
    MASK_RATIO = 0.2                # Mask 20% of observed values

    # All masking strategies to evaluate
    MASKING_STRATEGIES = [
        MaskingStrategy.MCAR,
        MaskingStrategy.SEQUENCE_END,
        MaskingStrategy.VARIABLE_WISE
    ]

    # All imputation methods to evaluate
    IMPUTATION_METHODS = [
        ImputationMethod.SMOOTHING_SPLINE,
        ImputationMethod.GAUSSIAN_PROCESS,
        ImputationMethod.MICE,
        ImputationMethod.HMM
    ]

    # =========================================================================
    # LOAD DATA
    # =========================================================================

    if TUNE:
        # Load combined set-a + set-b, split into train/test
        print("=" * 60)
        print("MODE: HYPERPARAMETER TUNING + TEST EVALUATION")
        print("=" * 60)

        train_data, train_patient_ids, test_data, test_patient_ids, features_info = load_combined_data(
            data_dir=DATA_DIR,
            missingness_threshold=MISSINGNESS_THRESHOLD,
            remove_outliers=REMOVE_OUTLIERS,
            normalize=NORMALIZE,
            test_ratio=TEST_RATIO,
            seed=SEED,
            verbose=True
        )

        all_results = []
        tuning_results = []

        # =====================================================================
        # RUN EXPERIMENTS WITH HYPERPARAMETER TUNING
        # =====================================================================

        for masking_strategy in MASKING_STRATEGIES:
            for method in IMPUTATION_METHODS:
                print(f"\n{'='*60}")
                print(f"Running: {masking_strategy.name} + {method.name}")
                print(f"{'='*60}")

                # Tune hyperparameters on training data using GroupKFold
                try:
                    best_params = tune_hyperparameters(
                        train_data=train_data,
                        train_patient_ids=train_patient_ids,
                        temporal_features=features_info['temporal_features'],
                        masking_strategy=masking_strategy,
                        imputation_method=method,
                        mask_ratio=MASK_RATIO,
                        seed=SEED,
                        n_splits=N_FOLDS
                    )

                    tuning_results.append({
                        'masking_strategy': masking_strategy.name,
                        'imputation_method': method.name,
                        'best_params': best_params
                    })

                    # Evaluate on test set with best parameters
                    print(f"\n  Evaluating on test set with best parameters...")

                    # Mask test data
                    masked_test = mask_dataset(
                        test_data,
                        strategy=masking_strategy,
                        features=features_info['temporal_features'],
                        mask_ratio=MASK_RATIO,
                        seed=SEED + 1000,
                        pivot_fn=pivot_timeseries
                    )

                    # Impute with best parameters
                    imputed_test = impute_patient_timeseries(
                        masked_test,
                        method=method,
                        **best_params
                    )

                    # Evaluate
                    test_metrics = evaluate_patient_imputation(masked_test, imputed_test)

                    # Store results
                    result_dict = {
                        'masking_strategy': masking_strategy.name,
                        'imputation_method': method.name,
                        'best_params': str(best_params),
                        'test_rmse': test_metrics.overall.rmse,
                        'test_mae': test_metrics.overall.mae,
                        'test_r2': test_metrics.overall.r2,
                        'test_mse': test_metrics.overall.mse,
                        'n_evaluated': test_metrics.overall.n_evaluated
                    }
                    all_results.append(result_dict)

                    print(f"\n  Test Results:")
                    print(f"    RMSE: {test_metrics.overall.rmse:.4f}")
                    print(f"    MAE:  {test_metrics.overall.mae:.4f}")
                    print(f"    R²:   {test_metrics.overall.r2:.4f}")
                    print(f"    n:    {test_metrics.overall.n_evaluated}")

                except ImportError as exc:
                    print(f"  Skipping method '{method.value}': {exc}")
                    continue
                except Exception as exc:
                    print(f"  Error with {masking_strategy.name} + {method.name}: {exc}")
                    continue

        # Save results
        results_df = pd.DataFrame(all_results)
        results_df.to_excel(OUTPUT_DIR / "tuned_results_test_set.xlsx", index=False)
        print(f"\n{'='*60}")
        print(f"Results saved to {OUTPUT_DIR / 'tuned_results_test_set.xlsx'}")
        print(f"{'='*60}")

        # Print summary table
        print("\n" + "=" * 60)
        print("FINAL TEST SET RESULTS")
        print("=" * 60)
        print(results_df.to_string(index=False))

    else:
        # Original workflow: use set-a only with train/val split
        print("=" * 60)
        print("MODE: VALIDATION SET EVALUATION (NO TUNING)")
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

        all_results = []

        for masking_strategy in MASKING_STRATEGIES:
            for method in IMPUTATION_METHODS:
                print(f"\n{'='*60}")
                print(f"Running: {masking_strategy.name} + {method.name}")
                print(f"{'='*60}")

                config = ExperimentConfig(
                    masking_strategy=masking_strategy,
                    imputation_method=method,
                    mask_ratio=MASK_RATIO,
                    seed=SEED
                )

                # Ensure method-specific output directory exists
                method_output_dir = OUTPUT_DIR / f"{masking_strategy.name}_{method.name}"
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

        # Save comparison metrics
        from experiment import results_to_dataframe
        summary_df = results_to_dataframe(all_results, 'val')
        summary_df.to_excel(OUTPUT_DIR / "imputation_comparison_summary.xlsx", index=False)
        print(f"\nComparison summary saved to {OUTPUT_DIR / 'imputation_comparison_summary.xlsx'}")

        # Print summary table
        print("\n" + "=" * 60)
        print("FINAL RESULTS SUMMARY")
        print("=" * 60)
        print_results_table(all_results, split='val')


if __name__ == "__main__":
    # Run experiments for all imputation methods
    main()
