"""
DataLoader for PhysioNet 2012 Challenge Dataset
Handles loading, preprocessing, train-val-test splitting, feature filtering,
normalization, and outlier removal.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


# All time-series variables from PhysioNet 2012
ALL_TIME_SERIES_VARIABLES = [
    'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Cholesterol',
    'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT',
    'HR', 'K', 'Lactate', 'Mg', 'MAP', 'MechVent', 'Na', 'NIDiasABP',
    'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
    'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC', 'Weight'
]

# Temporal features that should be predicted/imputed (continuous vital signs)
TEMPORAL_FEATURES = [
    'DiasABP', 'HR', 'MAP', 'SysABP', 'Urine',  # Will be filtered by missingness
    'NIDiasABP', 'NIMAP', 'NISysABP', 'RespRate', 'Temp', 'SaO2',
    'Glucose', 'BUN', 'Creatinine', 'HCT', 'HCO3', 'K', 'Na', 'pH',
    'PaCO2', 'PaO2', 'Lactate', 'WBC', 'Platelets'
]

# Static/categorical descriptors (collected at admission, used as input but NOT predicted)
STATIC_FEATURES = ['Age', 'Gender', 'Height', 'ICUType', 'Weight']
GENERAL_DESCRIPTORS = ['RecordID', 'Age', 'Gender', 'Height', 'ICUType', 'Weight']


@dataclass
class DataSplit:
    """Container for a data split with patient IDs and time series."""
    patient_ids: List[int]
    timeseries: Dict[int, pd.DataFrame]
    general_info: pd.DataFrame

    @property
    def n_patients(self) -> int:
        return len(self.patient_ids)


@dataclass
class PhysioNetData:
    """Container for train/val/test splits with metadata."""
    train: DataSplit
    val: DataSplit
    test: DataSplit
    features: List[str]              # All features used as input
    temporal_features: List[str]     # Features to predict/impute
    static_features: List[str]       # Features used as input only (not predicted)
    dropped_features: List[str]
    missingness_threshold: float
    scaler_params: Dict[str, Tuple[float, float]] = field(default_factory=dict)


def parse_time(time_str: str) -> float:
    """Convert HH:MM to hours as float."""
    h, m = map(int, time_str.split(':'))
    return h + m / 60.0


def load_patient_file(filepath: Path) -> Tuple[Dict, pd.DataFrame]:
    """Load single patient file, return general info and time series."""
    df = pd.read_csv(filepath)
    general_info = {}
    ts_rows = []

    for _, row in df.iterrows():
        t = parse_time(row['Time'])
        param, val = row['Parameter'], row['Value']

        if t == 0 and param in GENERAL_DESCRIPTORS:
            general_info[param] = val
        else:
            ts_rows.append({'time_hours': t, 'variable': param, 'value': val})

    return general_info, pd.DataFrame(ts_rows)


def load_raw_dataset(data_dir: Path, dataset_name: str,
                     verbose: bool = True) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
    """Load all patient files from a dataset directory."""
    dataset_path = data_dir / dataset_name
    files = sorted(dataset_path.glob('*.txt'))

    if verbose:
        print(f"Loading {len(files)} patients from {dataset_name}...")

    general_list = []
    timeseries = {}

    for i, f in enumerate(files):
        info, ts = load_patient_file(f)
        rid = int(info.get('RecordID', f.stem))
        general_list.append(info)
        timeseries[rid] = ts

        if verbose and (i + 1) % 1000 == 0:
            print(f"  Loaded {i + 1} patients...")

    if verbose:
        print(f"  Done. {len(files)} patients loaded.")

    return pd.DataFrame(general_list), timeseries


def pivot_timeseries(ts: pd.DataFrame) -> pd.DataFrame:
    """Convert long-format time series to wide format, replacing -1 with NaN."""
    if len(ts) == 0:
        return pd.DataFrame()

    pivoted = ts.pivot_table(
        index='time_hours', columns='variable', values='value', aggfunc='mean'
    )
    return pivoted.replace(-1, np.nan)


def compute_feature_missingness(timeseries: Dict[int, pd.DataFrame],
                                 features: List[str]) -> Dict[str, float]:
    """Compute missing rate for each feature across all patients."""
    counts = {f: {'obs': 0, 'total': 0} for f in features}

    for ts in timeseries.values():
        if len(ts) == 0:
            continue
        pivoted = pivot_timeseries(ts)
        n_rows = len(pivoted)

        for f in features:
            if f in pivoted.columns:
                counts[f]['obs'] += pivoted[f].notna().sum()
            counts[f]['total'] += n_rows

    return {
        f: 1 - (c['obs'] / c['total']) if c['total'] > 0 else 1.0
        for f, c in counts.items()
    }


def filter_features_by_missingness(timeseries: Dict[int, pd.DataFrame],
                                    threshold: float = 0.6) -> Tuple[List[str], List[str]]:
    """Filter features based on missingness threshold."""
    missingness = compute_feature_missingness(timeseries, ALL_TIME_SERIES_VARIABLES)

    kept = [f for f, rate in missingness.items() if rate <= threshold]
    dropped = [f for f, rate in missingness.items() if rate > threshold]

    return sorted(kept), sorted(dropped)


def filter_patient_timeseries(timeseries: Dict[int, pd.DataFrame],
                               features: List[str]) -> Dict[int, pd.DataFrame]:
    """Filter time series to only include specified features."""
    filtered = {}
    for pid, ts in timeseries.items():
        if len(ts) == 0:
            filtered[pid] = ts
        else:
            filtered[pid] = ts[ts['variable'].isin(features)]
    return filtered


def remove_outliers_iqr(timeseries: Dict[int, pd.DataFrame],
                        features: List[str],
                        iqr_multiplier: float = 3.0) -> Tuple[Dict[int, pd.DataFrame], Dict[str, Tuple[float, float]], int]:
    """Remove outliers using IQR method across the entire dataset."""
    all_values = {f: [] for f in features}

    for ts in timeseries.values():
        if len(ts) == 0:
            continue
        for _, row in ts.iterrows():
            var = row['variable']
            val = row['value']
            if var in features and val != -1 and not pd.isna(val):
                all_values[var].append(val)

    bounds = {}
    for f in features:
        if len(all_values[f]) > 0:
            q1 = np.percentile(all_values[f], 25)
            q3 = np.percentile(all_values[f], 75)
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            bounds[f] = (lower, upper)
        else:
            bounds[f] = (-np.inf, np.inf)

    filtered = {}
    outlier_count = 0

    for pid, ts in timeseries.items():
        if len(ts) == 0:
            filtered[pid] = ts
            continue

        ts_copy = ts.copy()
        for idx, row in ts_copy.iterrows():
            var = row['variable']
            val = row['value']
            if var in bounds and val != -1 and not pd.isna(val):
                lower, upper = bounds[var]
                if val < lower or val > upper:
                    ts_copy.at[idx, 'value'] = np.nan
                    outlier_count += 1

        filtered[pid] = ts_copy

    return filtered, bounds, outlier_count


def compute_normalization_params(timeseries: Dict[int, pd.DataFrame],
                                  features: List[str]) -> Dict[str, Tuple[float, float]]:
    """Compute mean and std for each feature from training data."""
    all_values = {f: [] for f in features}

    for ts in timeseries.values():
        if len(ts) == 0:
            continue
        for _, row in ts.iterrows():
            var = row['variable']
            val = row['value']
            if var in features and val != -1 and not pd.isna(val):
                all_values[var].append(val)

    params = {}
    for f in features:
        if len(all_values[f]) > 0:
            mean = np.mean(all_values[f])
            std = np.std(all_values[f])
            if std == 0:
                std = 1.0
            params[f] = (mean, std)
        else:
            params[f] = (0.0, 1.0)

    return params


def normalize_timeseries(timeseries: Dict[int, pd.DataFrame],
                          params: Dict[str, Tuple[float, float]]) -> Dict[int, pd.DataFrame]:
    """Normalize time series using precomputed mean/std."""
    normalized = {}

    for pid, ts in timeseries.items():
        if len(ts) == 0:
            normalized[pid] = ts
            continue

        ts_copy = ts.copy()
        for idx, row in ts_copy.iterrows():
            var = row['variable']
            val = row['value']
            if var in params and val != -1 and not pd.isna(val):
                mean, std = params[var]
                ts_copy.at[idx, 'value'] = (val - mean) / std

        normalized[pid] = ts_copy

    return normalized


def train_val_split(patient_ids: List[int],
                    val_ratio: float = 0.2,
                    seed: int = 42) -> Tuple[List[int], List[int]]:
    """Split patient IDs into train and validation sets."""
    rng = np.random.default_rng(seed)
    ids = np.array(patient_ids)
    rng.shuffle(ids)

    n_val = int(len(ids) * val_ratio)
    return ids[n_val:].tolist(), ids[:n_val].tolist()


def subset_data(patient_ids: List[int],
                timeseries: Dict[int, pd.DataFrame],
                general_info: pd.DataFrame) -> DataSplit:
    """Create a DataSplit for a subset of patients."""
    ts_subset = {pid: timeseries[pid] for pid in patient_ids}
    info_subset = general_info[general_info['RecordID'].isin(patient_ids)].reset_index(drop=True)
    return DataSplit(patient_ids=patient_ids, timeseries=ts_subset, general_info=info_subset)


def timeseries_to_flat_dataframe(timeseries: Dict[int, pd.DataFrame],
                                  features: List[str]) -> pd.DataFrame:
    """Convert patient timeseries dict to a flat DataFrame for saving."""
    rows = []

    for pid, ts in timeseries.items():
        if len(ts) == 0:
            continue
        pivoted = pivot_timeseries(ts)

        for t in pivoted.index:
            row = {'patient_id': pid, 'time_hours': t}
            for f in features:
                if f in pivoted.columns:
                    row[f] = pivoted.loc[t, f]
                else:
                    row[f] = np.nan
            rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values(['patient_id', 'time_hours']).reset_index(drop=True)


def save_split_to_xlsx(split: DataSplit, features: List[str],
                       filepath: Path, sheet_name: str = 'data') -> None:
    """Save a data split to an Excel file."""
    df = timeseries_to_flat_dataframe(split.timeseries, features)
    df.to_excel(filepath, sheet_name=sheet_name, index=False)


def load_physionet_data(data_dir: str = "data",
                        val_ratio: float = 0.2,
                        missingness_threshold: float = 0.6,
                        seed: int = 42,
                        remove_outliers: bool = True,
                        normalize: bool = True,
                        save_xlsx: bool = False,
                        output_dir: Optional[str] = None,
                        verbose: bool = True) -> PhysioNetData:
    """
    Load PhysioNet 2012 data with preprocessing, splits, and optional saving.
    """
    data_path = Path(data_dir)

    if output_dir is None:
        output_dir = Path(data_dir).parent / "output"
    else:
        output_dir = Path(output_dir)

    if save_xlsx:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data (set-a)
    if verbose:
        print("=" * 60)
        print("Loading PhysioNet 2012 Dataset")
        print("=" * 60)

    train_info, train_ts = load_raw_dataset(data_path, 'set-a', verbose)

    # Compute feature missingness on training data and filter
    if verbose:
        print(f"\nFiltering features (missingness threshold: {missingness_threshold:.0%})...")

    kept_features, dropped_features = filter_features_by_missingness(
        train_ts, missingness_threshold
    )

    # Determine which are temporal (to predict) vs all features
    temporal_features = [f for f in kept_features if f in TEMPORAL_FEATURES]

    if verbose:
        print(f"  Kept {len(kept_features)} features, dropped {len(dropped_features)} features")
        print(f"  Temporal features (to predict): {temporal_features}")

    # Filter training time series
    train_ts_filtered = filter_patient_timeseries(train_ts, kept_features)

    # Remove outliers from training data
    outlier_bounds = None
    if remove_outliers:
        if verbose:
            print("\nRemoving outliers (IQR method)...")
        train_ts_filtered, outlier_bounds, n_outliers = remove_outliers_iqr(
            train_ts_filtered, kept_features
        )
        if verbose:
            print(f"  Removed {n_outliers} outlier values")

    # Compute normalization parameters from training data
    scaler_params = {}
    if normalize:
        if verbose:
            print("\nComputing normalization parameters...")
        scaler_params = compute_normalization_params(train_ts_filtered, kept_features)
        train_ts_filtered = normalize_timeseries(train_ts_filtered, scaler_params)
        if verbose:
            print(f"  Normalized {len(kept_features)} features")

    # Split training into train/val
    all_train_ids = list(train_ts_filtered.keys())
    train_ids, val_ids = train_val_split(all_train_ids, val_ratio, seed)

    train_split = subset_data(train_ids, train_ts_filtered, train_info)
    val_split = subset_data(val_ids, train_ts_filtered, train_info)

    # Load test data (set-b)
    test_info, test_ts = load_raw_dataset(data_path, 'set-b', verbose)
    test_ts_filtered = filter_patient_timeseries(test_ts, kept_features)

    # Apply same outlier removal bounds to test data
    if remove_outliers and outlier_bounds is not None:
        test_ts_cleaned = {}
        for pid, ts in test_ts_filtered.items():
            if len(ts) == 0:
                test_ts_cleaned[pid] = ts
                continue
            ts_copy = ts.copy()
            for idx, row in ts_copy.iterrows():
                var = row['variable']
                val = row['value']
                if var in outlier_bounds and val != -1 and not pd.isna(val):
                    lower, upper = outlier_bounds[var]
                    if val < lower or val > upper:
                        ts_copy.at[idx, 'value'] = np.nan
            test_ts_cleaned[pid] = ts_copy
        test_ts_filtered = test_ts_cleaned

    # Apply same normalization to test data
    if normalize:
        test_ts_filtered = normalize_timeseries(test_ts_filtered, scaler_params)

    test_ids = list(test_ts_filtered.keys())
    test_split = subset_data(test_ids, test_ts_filtered, test_info)

    # Save to Excel if requested
    if save_xlsx:
        if verbose:
            print(f"\nSaving splits to {output_dir}...")

        train_val_ts = {**train_split.timeseries, **val_split.timeseries}
        train_val_info = pd.concat([train_split.general_info, val_split.general_info])
        train_val_combined = DataSplit(
            patient_ids=train_split.patient_ids + val_split.patient_ids,
            timeseries=train_val_ts,
            general_info=train_val_info
        )

        save_split_to_xlsx(train_val_combined, kept_features,
                          output_dir / "train_val_data.xlsx")
        save_split_to_xlsx(test_split, kept_features,
                          output_dir / "test_data.xlsx")

        if verbose:
            print(f"  Saved train_val_data.xlsx and test_data.xlsx")

    return PhysioNetData(
        train=train_split,
        val=val_split,
        test=test_split,
        features=kept_features,
        temporal_features=temporal_features,
        static_features=[f for f in kept_features if f not in temporal_features],
        dropped_features=dropped_features,
        missingness_threshold=missingness_threshold,
        scaler_params=scaler_params
    )


def print_data_summary(data: PhysioNetData) -> None:
    """Print summary of loaded data."""
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)

    print(f"\n{'Split':<12} {'Patients':>10}")
    print("-" * 24)
    print(f"{'Train':<12} {data.train.n_patients:>10}")
    print(f"{'Validation':<12} {data.val.n_patients:>10}")
    print(f"{'Test':<12} {data.test.n_patients:>10}")
    print("-" * 24)
    print(f"{'Total':<12} {data.train.n_patients + data.val.n_patients + data.test.n_patients:>10}")

    print(f"\nFeatures Summary")
    print("-" * 40)
    print(f"Missingness threshold: {data.missingness_threshold:.0%}")
    print(f"All features ({len(data.features)}): {data.features}")
    print(f"Temporal features to predict ({len(data.temporal_features)}): {data.temporal_features}")
    if data.static_features:
        print(f"Static features (input only) ({len(data.static_features)}): {data.static_features}")

    if data.scaler_params:
        print(f"\nNormalization parameters (mean, std):")
        for f, (mean, std) in data.scaler_params.items():
            print(f"  {f}: ({mean:.2f}, {std:.2f})")


if __name__ == "__main__":
    DATA_DIR = Path(__file__).parent / "data"

    data = load_physionet_data(
        data_dir=DATA_DIR,
        val_ratio=0.2,
        missingness_threshold=0.6,
        seed=42,
        remove_outliers=True,
        normalize=True,
        save_xlsx=True,
        verbose=True
    )

    print_data_summary(data)
