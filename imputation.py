"""
Imputation Methods for Irregular Time Series

Implements three imputation approaches:
1. Smoothing Splines - Cubic spline interpolation
2. Gaussian Processes - GP regression with RBF kernel
3. MICE - Multiple Imputation by Chained Equations (Bayesian)
"""

import os
from contextlib import contextmanager
import numpy as np
import pandas as pd
from enum import Enum
from typing import Optional, Dict
from scipy.interpolate import UnivariateSpline, interp1d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
import joblib
from joblib import Parallel, delayed
from tqdm.auto import tqdm

try:
    from hmmlearn.hmm import GaussianHMM
    _HMMLEARN_AVAILABLE = True
except ImportError:
    _HMMLEARN_AVAILABLE = False

warnings.filterwarnings('ignore')


class ImputationMethod(Enum):
    """Available imputation methods."""
    SMOOTHING_SPLINE = "smoothing_spline"
    GAUSSIAN_PROCESS = "gaussian_process"
    MICE = "mice"
    HMM = "hmm"


def smoothing_spline_impute(data: pd.DataFrame,
                            smoothing_factor: Optional[float] = None) -> pd.DataFrame:
    """
    Impute missing values using smoothing splines.

    Fits a cubic smoothing spline to observed values for each variable,
    then evaluates at all time points to fill missing values.

    Args:
        data: DataFrame with time as index, variables as columns
        smoothing_factor: Spline smoothing parameter (None = automatic)

    Returns:
        DataFrame with missing values imputed
    """
    imputed = data.copy()
    times = data.index.values.astype(float)

    for col in data.columns:
        series = data[col]
        observed_mask = series.notna()
        missing_mask = series.isna()

        if missing_mask.sum() == 0:
            continue

        n_observed = observed_mask.sum()

        # Need at least 2 points for interpolation
        if n_observed < 2:
            if n_observed == 1:
                imputed[col] = series.fillna(series.dropna().iloc[0])
            continue

        t_obs = times[observed_mask]
        y_obs = series[observed_mask].values

        try:
            if n_observed < 4:
                # Use linear interpolation with constant extrapolation for few points
                interp_func = interp1d(t_obs, y_obs, kind='linear',
                                       bounds_error=False, fill_value=(y_obs[0], y_obs[-1]))
                y_pred = interp_func(times)
            else:
                # Use cubic spline with constant extrapolation
                try:
                    # Use scipy CubicSpline which extrapolates with constant values by default
                    from scipy.interpolate import CubicSpline
                    spline = CubicSpline(t_obs, y_obs, bc_type='natural', extrapolate=False)
                    y_pred = spline(times, extrapolate=False)
                    # Fill extrapolated areas with nearest boundary value
                    y_pred = np.where(np.isnan(y_pred),
                                    np.where(times < t_obs.min(), y_obs[0], y_obs[-1]),
                                    y_pred)
                except Exception:
                    # Fall back to linear with constant extrapolation
                    interp_func = interp1d(t_obs, y_obs, kind='linear',
                                          bounds_error=False, fill_value=(y_obs[0], y_obs[-1]))
                    y_pred = interp_func(times)

            # Only fill missing values
            imputed.loc[missing_mask, col] = y_pred[missing_mask.values]

        except Exception:
            # Ultimate fallback: linear interpolation then forward/backward fill
            imputed[col] = series.interpolate(method='linear').ffill().bfill()

    return imputed


def gaussian_process_impute(data: pd.DataFrame,
                            length_scale: float = 5.0,
                            noise_level: float = 0.1,
                            max_training_points: int = 100,
                            random_state: int = 42,
                            optimize_hyperparams: bool = False) -> pd.DataFrame:
    """
    Impute missing values using Gaussian Process regression.

    Fits a GP with RBF kernel to observed values for each variable,
    then predicts at missing time points.

    Args:
        data: DataFrame with time as index, variables as columns
        length_scale: Initial length scale for RBF kernel
        noise_level: Initial noise level

    Returns:
        DataFrame with missing values imputed
    """
    rng = np.random.default_rng(random_state)
    imputed = data.copy()
    times = data.index.values.astype(float).reshape(-1, 1)

    for col in data.columns:
        series = data[col]
        observed_mask = series.notna()
        missing_mask = series.isna()

        if missing_mask.sum() == 0:
            continue

        if observed_mask.sum() < 2:
            if observed_mask.sum() == 1:
                imputed[col] = series.fillna(series.dropna().iloc[0])
            continue

        t_obs = times[observed_mask.values]
        y_obs = series[observed_mask].values
        t_missing = times[missing_mask.values]

        try:
            # Normalize y for numerical stability
            y_mean = y_obs.mean()
            y_std = y_obs.std() if y_obs.std() > 0 else 1.0
            y_normalized = (y_obs - y_mean) / y_std

            # Downsample observed points if too many (GP training is cubic)
            t_obs_fit = t_obs
            y_norm_fit = y_normalized
            if max_training_points and len(y_obs) > max_training_points:
                subset_idx = rng.choice(len(y_obs), size=max_training_points, replace=False)
                t_obs_fit = t_obs[subset_idx]
                y_norm_fit = y_normalized[subset_idx]

            # Define kernel
            kernel = (
                ConstantKernel(1.0 if optimize_hyperparams else 1.0, constant_value_bounds=(1e-3, 1e3))
                * RBF(length_scale=length_scale, length_scale_bounds=(1e-1, 1e2))
            )

            optimizer = "fmin_l_bfgs_b" if optimize_hyperparams else None
            n_restarts = 2 if optimize_hyperparams else 0

            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=noise_level ** 2,
                optimizer=optimizer,
                n_restarts_optimizer=n_restarts,
                normalize_y=False,
                random_state=random_state
            )
            gp.fit(t_obs_fit, y_norm_fit)

            # Predict at missing points
            y_pred_normalized = gp.predict(t_missing)
            y_pred = y_pred_normalized * y_std + y_mean

            # Clip to observed data range + margin (3 std devs)
            # to prevent extreme predictions
            y_min = y_obs.min() - 3 * y_std
            y_max = y_obs.max() + 3 * y_std
            y_pred = np.clip(y_pred, y_min, y_max)

            imputed.loc[missing_mask, col] = y_pred

        except Exception:
            # Fall back to linear interpolation
            imputed[col] = series.interpolate(method='linear').ffill().bfill()

    return imputed


def hmm_impute(data: pd.DataFrame,
               n_states: int = 4,
               max_iter: int = 50,
               max_time_steps: Optional[int] = 300,
               random_state: int = 42) -> pd.DataFrame:
    """
    Impute missing values using a Gaussian Hidden Markov Model.

    The HMM is fit on the multivariate time series with missing entries
    temporarily filled via simple forward/backward fill. After fitting,
    latent state posteriors (gamma) are used to compute the expected
    observation at each time step:
        E[x_t | data] ≈ sum_k gamma_tk * mean_k

    Returns:
        DataFrame with missing values imputed
    """
    if not _HMMLEARN_AVAILABLE:
        raise ImportError("hmmlearn is required for HMM imputation. "
                          "Install via 'pip install hmmlearn'.")

    if data.isna().sum().sum() == 0:
        return data.copy()

    # Simple fill to create a complete matrix for HMM fitting
    filled = data.copy()
    filled = filled.bfill().ffill()

    # If still NaNs (all-NaN column), fill with column-wise zeros
    nan_cols = filled.columns[filled.isna().all()]
    for col in nan_cols:
        filled[col] = 0.0
    filled = filled.fillna(0.0)

    X = filled.values.astype(float)

    X_fit = X
    if max_time_steps is not None and len(X) > max_time_steps:
        # Downsample uniformly to cap the time steps used for fitting
        indices = np.linspace(0, len(X) - 1, max_time_steps, dtype=int)
        X_fit = X[indices]

    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=max_iter,
        tol=1e-2,
        random_state=random_state,
        verbose=False
    )
    hmm.fit(X_fit)

    def _normalize_probs(vec: np.ndarray) -> np.ndarray:
        vec = np.nan_to_num(vec.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        total = vec.sum()
        if total <= 0 or not np.isfinite(total):
            vec = np.full_like(vec, 1.0 / len(vec))
        else:
            vec = vec / total
        return vec

    hmm.startprob_ = _normalize_probs(hmm.startprob_)
    hmm.transmat_ = np.vstack([_normalize_probs(row) for row in hmm.transmat_])

    # Posterior state responsibilities gamma_tk
    _, gamma = hmm.score_samples(X)

    # Expected observation at each time: sum_k gamma_tk * mean_k
    means = hmm.means_  # shape (K, D)
    expected_X = gamma @ means  # (T, D)

    expected_df = pd.DataFrame(
        expected_X,
        index=data.index,
        columns=data.columns
    )

    missing_mask = data.isna()
    imputed = data.copy()
    imputed = imputed.mask(missing_mask, expected_df)

    return imputed


def mice_impute(data: pd.DataFrame,
                max_iter: int = 5,
                n_nearest_features: Optional[int] = None,
                max_rows: Optional[int] = 200,
                seed: int = 42) -> pd.DataFrame:
    """
    Impute missing values using MICE (Multiple Imputation by Chained Equations).

    Uses iterative imputation where each feature is modeled as a function
    of other features, iterating until convergence.

    Args:
        data: DataFrame with time as index, variables as columns
        max_iter: Maximum number of imputation iterations
        n_nearest_features: Number of features to use for imputation (None = all)
        seed: Random seed

    Returns:
        DataFrame with missing values imputed
    """
    if data.isna().sum().sum() == 0:
        return data.copy()

    if len(data) < 2 or data.shape[1] < 1:
        return data.copy()

    rng = np.random.default_rng(seed)
    n_features = data.shape[1]
    if n_nearest_features is None:
        n_nearest = min(5, n_features)
    else:
        n_nearest = min(n_nearest_features, n_features)

    try:
        imputer = IterativeImputer(
            max_iter=max_iter,
            n_nearest_features=n_nearest,
            random_state=seed,
            initial_strategy='mean',
            skip_complete=True,
            sample_posterior=False
        )

        fit_values = data.values
        if max_rows is not None and len(data) > max_rows:
            subset_idx = rng.choice(len(data), size=max_rows, replace=False)
            fit_values = data.iloc[subset_idx].values

        imputer.fit(fit_values)
        imputed_values = imputer.transform(data.values)
        imputed = pd.DataFrame(
            imputed_values,
            index=data.index,
            columns=data.columns
        )
    except Exception:
        # Fall back to simple mean imputation
        imputed = data.fillna(data.mean())
        # If still NaN (all column was NaN), fill with 0
        imputed = imputed.fillna(0)

    return imputed


def impute(data: pd.DataFrame, method: ImputationMethod, **kwargs):
    """
    Apply an imputation method to data.

    Returns:
        DataFrame with missing values imputed
    """
    if method == ImputationMethod.SMOOTHING_SPLINE:
        return smoothing_spline_impute(data, **kwargs)
    elif method == ImputationMethod.GAUSSIAN_PROCESS:
        return gaussian_process_impute(data, **kwargs)
    elif method == ImputationMethod.MICE:
        return mice_impute(data, **kwargs)
    elif method == ImputationMethod.HMM:
        return hmm_impute(data, **kwargs)
    else:
        raise ValueError(f"Unknown imputation method: {method}")


@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bars."""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


def _impute_single_patient(pid, mdata, method, kwargs):
    df = mdata.data if hasattr(mdata, 'data') else mdata
    return pid, impute(df, method, **kwargs)


def impute_patient_timeseries(masked_data: Dict,
                               method: ImputationMethod,
                               **kwargs) -> Dict[int, pd.DataFrame]:
    """
    Apply imputation to all patients' masked time series.

    Returns:
        Dict[patient_id, imputed DataFrame]
    """
    imputed_data = {}
    if not masked_data:
        return imputed_data

    show_progress = kwargs.pop('show_progress', True)
    progress_desc = kwargs.pop('progress_desc', f"{method.value} imputation")

    n_jobs = kwargs.pop('n_jobs', None)
    if n_jobs is None:
        if method == ImputationMethod.GAUSSIAN_PROCESS:
            cpu_count = os.cpu_count() or 1
            n_jobs = max(1, min(cpu_count, 4))
        else:
            n_jobs = 1
    else:
        n_jobs = max(1, n_jobs)

    items = list(masked_data.items())
    n_jobs = min(n_jobs, len(items))

    if n_jobs == 1:
        iterator = tqdm(items, desc=progress_desc, unit="patient") if show_progress else items
        for pid, mdata in iterator:
            pid_result, df = _impute_single_patient(pid, mdata, method, kwargs)
            imputed_data[pid_result] = df
        if show_progress:
            iterator.close()
    else:
        if show_progress:
            with tqdm_joblib(tqdm(total=len(items), desc=progress_desc, unit="patient")):
                results = Parallel(n_jobs=n_jobs)(
                    delayed(_impute_single_patient)(pid, mdata, method, kwargs)
                    for pid, mdata in items
                )
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(_impute_single_patient)(pid, mdata, method, kwargs)
                for pid, mdata in items
            )
        imputed_data = {pid: df for pid, df in results}

    return imputed_data


if __name__ == "__main__":
    # Demo/test
    np.random.seed(42)

    # Create sample data with missing values
    times = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    data = pd.DataFrame({
        'HR': [72, 75, np.nan, 80, np.nan, 76, 74, 73],
        'Temp': [36.5, np.nan, 37.0, 37.2, np.nan, 36.8, np.nan, 36.9],
        'BP': [120, 118, np.nan, np.nan, 116, 115, 117, 118]
    }, index=times)
    data.index.name = 'time_hours'

    print("Original Data with Missing Values:")
    print(data)
    print(f"\nMissing values: {data.isna().sum().sum()}")

    for method in ImputationMethod:
        print(f"\n{'='*50}")
        print(f"Method: {method.value}")
        print('='*50)

        try:
            imputed = impute(data, method)
        except ImportError as e:
            print(f"Skipping method '{method.value}': {e}")
            continue

        print(f"\nImputed Data:")
        print(imputed.round(2))
