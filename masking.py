"""
Masking Strategies for Missing Value Imputation Evaluation

Implements three masking strategies:
- MCAR: Dataset-wide random masking
- SEQUENCE_END: Patient-wise sequence end masking
- VARIABLE_WISE: Dataset-wide variable masking based on missingness
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import Tuple, Dict, List
from dataclasses import dataclass


class MaskingStrategy(Enum):
    """Available masking strategies."""
    MCAR = "mcar"                    # Missing Completely At Random (dataset-wide)
    SEQUENCE_END = "sequence_end"    # Mask end of sequences (patient-wise)
    VARIABLE_WISE = "variable_wise"  # Mask high-missingness variables (dataset-wide)


@dataclass
class MaskedData:
    """Container for masked data with tracking information."""
    data: pd.DataFrame          # Data with artificial missing values (NaN)
    mask: pd.DataFrame          # Boolean mask: True = artificially masked
    original: pd.DataFrame      # Original data before masking

    @property
    def n_masked(self) -> int:
        """Number of artificially masked values."""
        return self.mask.sum().sum()

    @property
    def n_observed_original(self) -> int:
        """Number of observed values in original data."""
        return self.original.notna().sum().sum()

    @property
    def mask_ratio_actual(self) -> float:
        """Actual ratio of masked values to original observed values."""
        if self.n_observed_original == 0:
            return 0.0
        return self.n_masked / self.n_observed_original


@dataclass
class DatasetMaskedData:
    """Container for dataset-wide masked data."""
    patient_data: Dict[int, MaskedData]  # Per-patient masked data
    global_mask_indices: List[Tuple[int, float, str]]  # (patient_id, time, variable)


def sequence_end_mask_patient(data: pd.DataFrame, mask_ratio: float = 0.2) -> MaskedData:
    """
    Sequence-end masking for a single patient (patient-wise).

    Masks the last mask_ratio proportion of observed time points for each variable.
    """
    mask = pd.DataFrame(False, index=data.index, columns=data.columns)

    for col in data.columns:
        observed_idx = data[col].dropna().index
        if len(observed_idx) == 0:
            continue

        n_to_mask = max(1, int(len(observed_idx) * mask_ratio))
        indices_to_mask = observed_idx[-n_to_mask:]
        mask.loc[indices_to_mask, col] = True

    masked_data = data.copy()
    masked_data[mask] = np.nan

    return MaskedData(data=masked_data, mask=mask, original=data.copy())


def create_dataset_wide_mask_mcar(flat_df: pd.DataFrame,
                                   features: List[str],
                                   mask_ratio: float = 0.2,
                                   seed: int = 42) -> pd.DataFrame:
    """
    Create MCAR mask across the entire dataset.

    Args:
        flat_df: Flat DataFrame with patient_id, time_hours, and feature columns
        features: List of feature names
        mask_ratio: Proportion of observed values to mask
        seed: Random seed

    Returns:
        DataFrame with mask columns (True = mask this value)
    """
    rng = np.random.default_rng(seed)

    # Collect all observed positions
    observed_positions = []
    for idx, row in flat_df.iterrows():
        for f in features:
            if f in flat_df.columns and pd.notna(row[f]):
                observed_positions.append((idx, f))

    # Randomly select positions to mask
    n_to_mask = int(len(observed_positions) * mask_ratio)
    mask_indices = rng.choice(len(observed_positions), size=n_to_mask, replace=False)

    # Create mask DataFrame
    mask_df = flat_df[['patient_id', 'time_hours']].copy()
    for f in features:
        mask_df[f] = False

    for i in mask_indices:
        idx, f = observed_positions[i]
        mask_df.at[idx, f] = True

    return mask_df


def create_dataset_wide_mask_variable_wise(flat_df: pd.DataFrame,
                                            features: List[str],
                                            mask_ratio: float = 0.2,
                                            seed: int = 42) -> pd.DataFrame:
    """
    Create variable-wise mask across the entire dataset.

    Masks entire variables starting from those with highest natural missingness
    until approximately mask_ratio of total observed values are masked.

    Args:
        flat_df: Flat DataFrame with patient_id, time_hours, and feature columns
        features: List of feature names
        mask_ratio: Target proportion of observed values to mask
        seed: Random seed for tie-breaking

    Returns:
        DataFrame with mask columns
    """
    rng = np.random.default_rng(seed)

    # Calculate missingness and observed count for each variable
    var_stats = {}
    for f in features:
        if f in flat_df.columns:
            n_observed = flat_df[f].notna().sum()
            n_total = len(flat_df)
            miss_rate = 1 - (n_observed / n_total) if n_total > 0 else 1.0
            var_stats[f] = {'miss_rate': miss_rate, 'n_observed': n_observed}

    # Sort by missingness (highest first)
    variables = list(var_stats.keys())
    rng.shuffle(variables)
    variables_sorted = sorted(variables, key=lambda x: var_stats[x]['miss_rate'], reverse=True)

    # Calculate target
    total_observed = sum(var_stats[f]['n_observed'] for f in features if f in var_stats)
    target_masked = int(total_observed * mask_ratio)

    # Select variables to mask
    mask_df = flat_df[['patient_id', 'time_hours']].copy()
    for f in features:
        mask_df[f] = False

    masked_count = 0
    for var in variables_sorted:
        if masked_count >= target_masked:
            break

        # Mask all observed values in this variable
        observed_mask = flat_df[var].notna()
        mask_df.loc[observed_mask, var] = True
        masked_count += var_stats[var]['n_observed']

    return mask_df


def apply_mask_to_patients(flat_df: pd.DataFrame,
                           mask_df: pd.DataFrame,
                           features: List[str]) -> Dict[int, MaskedData]:
    """
    Apply a dataset-wide mask to individual patient data.

    Args:
        flat_df: Original flat DataFrame
        mask_df: Mask DataFrame (same shape as flat_df)
        features: List of feature names

    Returns:
        Dict mapping patient_id to MaskedData
    """
    patient_data = {}

    for pid in flat_df['patient_id'].unique():
        # Get patient rows
        patient_rows = flat_df[flat_df['patient_id'] == pid].copy()
        mask_rows = mask_df[mask_df['patient_id'] == pid].copy()

        if len(patient_rows) == 0:
            continue

        # Convert to wide format
        original = patient_rows.set_index('time_hours')[features].copy()
        mask = mask_rows.set_index('time_hours')[features].copy()

        # Apply mask
        masked = original.copy()
        masked[mask] = np.nan

        patient_data[pid] = MaskedData(
            data=masked,
            mask=mask.astype(bool),
            original=original
        )

    return patient_data


def mask_dataset(timeseries: Dict[int, pd.DataFrame],
                 strategy: MaskingStrategy,
                 features: List[str],
                 mask_ratio: float = 0.2,
                 seed: int = 42,
                 pivot_fn=None,
                 general_info: pd.DataFrame = None) -> Dict[int, MaskedData]:
    """
    Apply masking strategy to entire dataset.

    Args:
        timeseries: Dict mapping patient_id to time series DataFrame (long format)
        strategy: Masking strategy to apply
        features: List of feature names (temporal features to mask)
        mask_ratio: Proportion to mask
        seed: Random seed
        pivot_fn: Function to convert long to wide format
        general_info: DataFrame with static features (Age, Gender, Height, ICUType)

    Returns:
        Dict mapping patient_id to MaskedData
    """
    # Helper to get patient info
    def get_patient_info(pid):
        if general_info is not None and len(general_info) > 0:
            patient_row = general_info[general_info['RecordID'] == pid]
            if len(patient_row) > 0:
                return patient_row.iloc[0].to_dict()
        return None

    if strategy == MaskingStrategy.SEQUENCE_END:
        # Patient-wise masking
        masked_data = {}
        for i, (pid, ts) in enumerate(timeseries.items()):
            if pivot_fn is not None:
                patient_info = get_patient_info(pid)
                wide_ts = pivot_fn(ts, patient_info)
            else:
                wide_ts = ts

            if len(wide_ts) == 0:
                masked_data[pid] = MaskedData(
                    data=wide_ts,
                    mask=pd.DataFrame(False, index=wide_ts.index, columns=wide_ts.columns),
                    original=wide_ts
                )
                continue

            # Mask only the temporal features (not Age, Gender, Height, ICUType)
            masked_data[pid] = sequence_end_mask_patient(wide_ts[features], mask_ratio)
            # Add back static features to the masked data
            for col in wide_ts.columns:
                if col not in features and col in wide_ts.columns:
                    masked_data[pid].data[col] = wide_ts[col]
                    masked_data[pid].original[col] = wide_ts[col]
                    masked_data[pid].mask[col] = False

        return masked_data

    else:
        # Dataset-wide masking (MCAR or VARIABLE_WISE)

        # First, convert all patient data to flat DataFrame
        rows = []
        for pid, ts in timeseries.items():
            if pivot_fn is not None:
                patient_info = get_patient_info(pid)
                wide_ts = pivot_fn(ts, patient_info)
            else:
                wide_ts = ts

            if len(wide_ts) == 0:
                continue

            # Extract all features (temporal + static)
            all_features = list(wide_ts.columns)

            for t in wide_ts.index:
                row = {'patient_id': pid, 'time_hours': t}
                for f in all_features:
                    if f in wide_ts.columns:
                        row[f] = wide_ts.loc[t, f]
                    else:
                        row[f] = np.nan
                rows.append(row)

        flat_df = pd.DataFrame(rows)

        if len(flat_df) == 0:
            return {}

        # Create dataset-wide mask (only for temporal features)
        if strategy == MaskingStrategy.MCAR:
            mask_df = create_dataset_wide_mask_mcar(flat_df, features, mask_ratio, seed)
        elif strategy == MaskingStrategy.VARIABLE_WISE:
            mask_df = create_dataset_wide_mask_variable_wise(flat_df, features, mask_ratio, seed)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Add columns for static features (never masked)
        static_features = [c for c in flat_df.columns if c not in features and c not in ['patient_id', 'time_hours']]
        for f in static_features:
            if f not in mask_df.columns:
                mask_df[f] = False

        # Apply mask to individual patients (now includes static features)
        all_features_to_apply = features + static_features
        return apply_mask_to_patients(flat_df, mask_df, all_features_to_apply)


# Legacy function for backward compatibility
def mask_patient_timeseries(timeseries: Dict[int, pd.DataFrame],
                            strategy: MaskingStrategy,
                            mask_ratio: float = 0.2,
                            seed: int = 42,
                            pivot_fn=None,
                            features: List[str] = None) -> Dict[int, MaskedData]:
    """
    Legacy wrapper for mask_dataset.
    """
    if features is None:
        # Infer features from first patient
        for ts in timeseries.values():
            if pivot_fn is not None:
                wide = pivot_fn(ts)
            else:
                wide = ts
            if len(wide) > 0:
                features = list(wide.columns)
                break

    if features is None:
        features = []

    return mask_dataset(timeseries, strategy, features, mask_ratio, seed, pivot_fn)


if __name__ == "__main__":
    # Demo/test
    np.random.seed(42)

    # Create sample data for 3 patients
    def create_patient_data(pid, seed):
        rng = np.random.default_rng(seed)
        times = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
        data = pd.DataFrame({
            'HR': rng.normal(75, 5, len(times)),
            'Temp': rng.normal(37, 0.5, len(times)),
            'BP': rng.normal(120, 10, len(times))
        }, index=times)
        data.index.name = 'time_hours'
        # Add some natural missing
        data.iloc[2, 0] = np.nan
        data.iloc[4, 1] = np.nan
        return data

    timeseries = {
        1: create_patient_data(1, 42),
        2: create_patient_data(2, 43),
        3: create_patient_data(3, 44)
    }

    features = ['HR', 'Temp', 'BP']

    print("Testing masking strategies on 3 patients\n")

    for strategy in MaskingStrategy:
        print(f"\n{'='*50}")
        print(f"Strategy: {strategy.value}")
        print('='*50)

        result = mask_dataset(timeseries, strategy, features, mask_ratio=0.3, seed=42)

        total_masked = sum(m.n_masked for m in result.values())
        total_observed = sum(m.n_observed_original for m in result.values())

        print(f"Total masked: {total_masked} / {total_observed} observed "
              f"({total_masked/total_observed:.1%})")

        for pid, mdata in result.items():
            print(f"\n  Patient {pid}: masked {mdata.n_masked} values")
