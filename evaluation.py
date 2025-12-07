"""
Evaluation Metrics for Imputation Quality

Computes MAE, MSE, RMSE, and R² between original and imputed values,
only at positions that were artificially masked.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ImputationMetrics:
    """Container for imputation evaluation metrics."""
    mae: float          # Mean Absolute Error
    mse: float          # Mean Squared Error
    rmse: float         # Root Mean Squared Error
    r2: float           # R-squared (coefficient of determination)
    n_evaluated: int    # Number of values evaluated

    def __repr__(self) -> str:
        return (f"ImputationMetrics(MAE={self.mae:.4f}, MSE={self.mse:.4f}, "
                f"RMSE={self.rmse:.4f}, R²={self.r2:.4f}, n={self.n_evaluated})")


@dataclass
class PatientMetrics:
    """Metrics aggregated across all patients."""
    overall: ImputationMetrics
    per_patient: Dict[int, ImputationMetrics] = field(default_factory=dict)
    per_variable: Dict[str, ImputationMetrics] = field(default_factory=dict)


def compute_metrics(original: pd.DataFrame,
                    imputed: pd.DataFrame,
                    mask: pd.DataFrame) -> ImputationMetrics:
    """
    Compute imputation metrics on masked positions only.

    Args:
        original: Original data before masking
        imputed: Data after imputation
        mask: Boolean mask where True = artificially masked position

    Returns:
        ImputationMetrics object
    """
    # Extract values at masked positions
    y_true = original.values[mask.values]
    y_pred = imputed.values[mask.values]

    # Filter out any remaining NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    n = len(y_true)

    if n == 0:
        return ImputationMetrics(
            mae=np.nan, mse=np.nan, rmse=np.nan, r2=np.nan, n_evaluated=0
        )

    # Compute metrics
    errors = y_pred - y_true
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)

    # R² = 1 - SS_res / SS_tot
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    return ImputationMetrics(mae=mae, mse=mse, rmse=rmse, r2=r2, n_evaluated=n)


def compute_metrics_per_variable(original: pd.DataFrame,
                                  imputed: pd.DataFrame,
                                  mask: pd.DataFrame) -> Dict[str, ImputationMetrics]:
    """
    Compute metrics for each variable separately.

    Args:
        original: Original data before masking
        imputed: Data after imputation
        mask: Boolean mask

    Returns:
        Dict mapping variable name to ImputationMetrics
    """
    metrics = {}

    for col in original.columns:
        col_mask = mask[col] if col in mask.columns else pd.Series(False, index=mask.index)

        if col_mask.sum() == 0:
            continue

        y_true = original[col].values[col_mask.values]
        y_pred = imputed[col].values[col_mask.values]

        # Filter NaN
        valid = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true, y_pred = y_true[valid], y_pred[valid]

        n = len(y_true)
        if n == 0:
            continue

        errors = y_pred - y_true
        mae = np.mean(np.abs(errors))
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)

        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        metrics[col] = ImputationMetrics(mae=mae, mse=mse, rmse=rmse, r2=r2, n_evaluated=n)

    return metrics


def evaluate_patient_imputation(masked_data: Dict,
                                 imputed_data: Dict[int, pd.DataFrame]) -> PatientMetrics:
    """
    Evaluate imputation across all patients.

    Args:
        masked_data: Dict mapping patient_id to MaskedData objects
        imputed_data: Dict mapping patient_id to imputed DataFrames

    Returns:
        PatientMetrics with overall and per-patient metrics
    """
    all_y_true = []
    all_y_pred = []
    per_patient = {}
    per_variable_agg = {}

    for pid in masked_data.keys():
        if pid not in imputed_data:
            continue

        mdata = masked_data[pid]
        imputed = imputed_data[pid]

        # Get original, mask from MaskedData
        if hasattr(mdata, 'original'):
            original = mdata.original
            mask = mdata.mask
        else:
            continue

        if mask.sum().sum() == 0:
            continue

        # Per-patient metrics
        per_patient[pid] = compute_metrics(original, imputed, mask)

        # Per-variable metrics aggregation
        var_metrics = compute_metrics_per_variable(original, imputed, mask)
        for var, m in var_metrics.items():
            if var not in per_variable_agg:
                per_variable_agg[var] = {'y_true': [], 'y_pred': []}

            y_true = original[var].values[mask[var].values]
            y_pred = imputed[var].values[mask[var].values]
            valid = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_valid = y_true[valid]
            y_pred_valid = y_pred[valid]

            per_variable_agg[var]['y_true'].extend(y_true_valid)
            per_variable_agg[var]['y_pred'].extend(y_pred_valid)

        # Aggregate for overall metrics
        y_true = original.values[mask.values]
        y_pred = imputed.values[mask.values]
        valid = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_valid = y_true[valid]
        y_pred_valid = y_pred[valid]

        all_y_true.extend(y_true_valid)
        all_y_pred.extend(y_pred_valid)

    # Compute overall metrics
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    if len(all_y_true) == 0:
        overall = ImputationMetrics(
            mae=np.nan, mse=np.nan, rmse=np.nan, r2=np.nan, n_evaluated=0
        )
    else:
        errors = all_y_pred - all_y_true
        mae = np.mean(np.abs(errors))
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((all_y_true - np.mean(all_y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        overall = ImputationMetrics(mae=mae, mse=mse, rmse=rmse, r2=r2,
                                    n_evaluated=len(all_y_true))

    # Compute per-variable metrics from aggregated data
    per_variable = {}
    for var, data in per_variable_agg.items():
        y_true = np.array(data['y_true'])
        y_pred = np.array(data['y_pred'])

        if len(y_true) == 0:
            continue

        errors = y_pred - y_true
        mae = np.mean(np.abs(errors))
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        per_variable[var] = ImputationMetrics(mae=mae, mse=mse, rmse=rmse,
                                               r2=r2, n_evaluated=len(y_true))

    return PatientMetrics(overall=overall, per_patient=per_patient,
                          per_variable=per_variable)


def summarize_results(metrics: PatientMetrics,
                      show_per_patient: bool = False) -> str:
    """
    Generate a summary string of evaluation results.

    Args:
        metrics: PatientMetrics object
        show_per_patient: Whether to include per-patient breakdown

    Returns:
        Formatted summary string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("IMPUTATION EVALUATION RESULTS")
    lines.append("=" * 60)

    # Overall metrics
    lines.append("\n--- Overall Metrics ---")
    lines.append(f"  MAE:  {metrics.overall.mae:.4f}")
    lines.append(f"  MSE:  {metrics.overall.mse:.4f}")
    lines.append(f"  RMSE: {metrics.overall.rmse:.4f}")
    lines.append(f"  R²:   {metrics.overall.r2:.4f}")
    lines.append(f"  N:    {metrics.overall.n_evaluated}")

    # Per-variable metrics
    if metrics.per_variable:
        lines.append("\n--- Per-Variable Metrics ---")
        lines.append(f"  {'Variable':<15} {'MAE':>10} {'RMSE':>10} {'R²':>10} {'N':>8}")
        lines.append("  " + "-" * 55)

        for var in sorted(metrics.per_variable.keys()):
            m = metrics.per_variable[var]
            r2_str = f"{m.r2:.4f}" if not np.isnan(m.r2) else "N/A"
            lines.append(f"  {var:<15} {m.mae:>10.4f} {m.rmse:>10.4f} {r2_str:>10} {m.n_evaluated:>8}")

    # Per-patient metrics (optional)
    if show_per_patient and metrics.per_patient:
        lines.append("\n--- Per-Patient Metrics (sample) ---")
        sample_pids = list(metrics.per_patient.keys())[:5]
        for pid in sample_pids:
            m = metrics.per_patient[pid]
            lines.append(f"  Patient {pid}: MAE={m.mae:.4f}, RMSE={m.rmse:.4f}, N={m.n_evaluated}")

    return "\n".join(lines)


if __name__ == "__main__":
    from masking import MaskedData

    # Demo/test
    np.random.seed(42)

    # Create sample data
    times = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    original = pd.DataFrame({
        'HR': [72, 75, 78, 80, 78, 76, 74],
        'Temp': [36.5, 36.7, 37.0, 37.2, 37.0, 36.8, 36.9],
    }, index=times)

    # Simulate mask
    mask = pd.DataFrame({
        'HR': [False, False, True, True, False, False, False],
        'Temp': [False, True, False, False, True, False, False],
    }, index=times)

    # Simulate imputed values (with some error)
    imputed = original.copy()
    imputed.loc[1.0, 'HR'] = 77.5    # True: 78
    imputed.loc[2.0, 'HR'] = 81.0    # True: 80
    imputed.loc[0.5, 'Temp'] = 36.6  # True: 36.7
    imputed.loc[3.0, 'Temp'] = 36.9  # True: 37.0

    print("Original:")
    print(original)
    print("\nMask (True = evaluate here):")
    print(mask)
    print("\nImputed:")
    print(imputed)

    metrics = compute_metrics(original, imputed, mask)
    print(f"\nMetrics: {metrics}")

    var_metrics = compute_metrics_per_variable(original, imputed, mask)
    print("\nPer-variable metrics:")
    for var, m in var_metrics.items():
        print(f"  {var}: {m}")
