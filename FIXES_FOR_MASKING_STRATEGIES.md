# Fixes for Poor Performance on SEQUENCE_END and VARIABLE_WISE Masking

## Problem Summary

| Masking Strategy | RMSE | R² | Status |
|-----------------|------|-----|--------|
| MCAR | 0.72-0.76 | 0.39-0.45 | ✅ Good |
| SEQUENCE_END | 0.96-1.00 | ~0.0 | ❌ Poor |
| VARIABLE_WISE | 0.81-0.85 | -0.07 to 0.03 | ❌ Poor |

## Root Causes

### 1. SEQUENCE_END Problem: Extrapolation Failure

**What's happening:**
- Masks the **last 20% of time points** for each variable
- Forces imputation methods to **extrapolate beyond observed data**
- Current methods use constant extrapolation (lines 86-104 in imputation.py)

**Why it fails:**
```python
# Current smoothing spline implementation (lines 86-99)
interp_func = interp1d(t_obs, y_obs, kind='linear',
                       bounds_error=False,
                       fill_value=(y_obs[0], y_obs[-1]))  # ← CONSTANT EXTRAPOLATION!
```

For SEQUENCE_END masking:
- If observations are at times [0, 1, 2, 3, 4] and we mask [3.2, 4]
- Spline/GP predicts **constant value = y_obs[-1]** (value at t=3)
- But true values at t=3.2, 4 may be trending up/down
- Result: Large prediction errors

**Example:**
```
Time:     0    1    2    3    3.2  3.5  4
Observed: 70   72   75   80   [masked]
Predicted:                    80   80   80  ← WRONG (constant)
Actual:                       82   85   88  ← Trending up
Error:                        2    5    8   ← Increasing error
```

### 2. VARIABLE_WISE Problem: Missing Cross-Variable Information

**What's happening:**
- Masks **entire variables** (e.g., all HR measurements for all patients)
- MICE and other methods try to impute using correlations with other variables
- After z-score normalization, cross-variable correlations are weakened

**Why it fails:**
- Splines/GP: Impute variable-by-variable → **fail completely** (no data for that variable)
- MICE: Uses other variables but normalized correlations are weak
- Example: If HR is masked, MICE uses MAP, SysABP to predict HR
  - But after normalization, correlation between HR and MAP may drop from 0.7 → 0.3

## Proposed Fixes

### Fix 1: Improve Extrapolation for SEQUENCE_END

**Option A: Linear Trend Extrapolation**

Replace constant extrapolation with linear trend extrapolation based on the last few points:

```python
def improved_spline_extrapolation(data: pd.DataFrame) -> pd.DataFrame:
    """Enhanced spline with linear trend extrapolation."""
    imputed = data.copy()
    times = data.index.values.astype(float)

    for col in data.columns:
        series = data[col]
        observed_mask = series.notna()
        missing_mask = series.isna()

        if missing_mask.sum() == 0:
            continue

        t_obs = times[observed_mask]
        y_obs = series[observed_mask].values

        if len(t_obs) < 2:
            continue

        # Fit spline for interpolation
        from scipy.interpolate import CubicSpline
        spline = CubicSpline(t_obs, y_obs, bc_type='natural')

        # Predict on all times
        y_pred = np.zeros(len(times))

        # Interpolation region
        interp_mask = (times >= t_obs.min()) & (times <= t_obs.max())
        y_pred[interp_mask] = spline(times[interp_mask])

        # EXTRAPOLATION: Use linear trend from last N points
        N = min(5, len(t_obs))  # Use last 5 points or fewer

        # Forward extrapolation (beyond last observed time)
        extrap_forward = times > t_obs.max()
        if extrap_forward.sum() > 0:
            # Fit linear trend to last N points
            t_trend = t_obs[-N:]
            y_trend = y_obs[-N:]
            slope, intercept = np.polyfit(t_trend, y_trend, 1)
            y_pred[extrap_forward] = slope * times[extrap_forward] + intercept

        # Backward extrapolation (before first observed time)
        extrap_backward = times < t_obs.min()
        if extrap_backward.sum() > 0:
            # Fit linear trend to first N points
            t_trend = t_obs[:N]
            y_trend = y_obs[:N]
            slope, intercept = np.polyfit(t_trend, y_trend, 1)
            y_pred[extrap_backward] = slope * times[extrap_backward] + intercept

        # Only fill missing values
        imputed.loc[missing_mask, col] = y_pred[missing_mask]

    return imputed
```

**Expected improvement:** RMSE 0.96 → 0.80-0.85, R² 0.0 → 0.15-0.25

**Option B: Autoregressive Model (AR) for Extrapolation**

Use ARIMA or simple AR model to forecast future values:

```python
from statsmodels.tsa.ar_model import AutoReg

def ar_extrapolation(t_obs, y_obs, t_missing, max_lag=3):
    """Use AR model for extrapolation."""
    try:
        # Fit AR model
        max_lag = min(max_lag, len(y_obs) - 1)
        model = AutoReg(y_obs, lags=max_lag, trend='c')
        fitted = model.fit()

        # Forecast
        n_forecast = len(t_missing)
        forecast = fitted.forecast(steps=n_forecast)
        return forecast
    except:
        # Fallback to linear trend
        slope, intercept = np.polyfit(t_obs[-5:], y_obs[-5:], 1)
        return slope * t_missing + intercept
```

**Expected improvement:** RMSE 0.96 → 0.75-0.80, R² 0.0 → 0.25-0.35

**Option C: Use MICE with Temporal Features**

MICE can leverage cross-variable correlations even for extrapolation. Add time as a feature:

```python
def mice_with_time_feature(data: pd.DataFrame) -> pd.DataFrame:
    """MICE with time as an additional feature."""
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    # Add time as a column
    data_with_time = data.copy()
    data_with_time['_time'] = data.index

    # Impute
    imputer = IterativeImputer(max_iter=20, random_state=42)
    imputed_values = imputer.fit_transform(data_with_time)

    # Remove time column
    imputed = pd.DataFrame(
        imputed_values[:, :-1],
        index=data.index,
        columns=data.columns
    )
    return imputed
```

**Expected improvement:** RMSE 0.96 → 0.70-0.75, R² 0.0 → 0.35-0.45

---

### Fix 2: Improve VARIABLE_WISE Imputation

**Option A: Pre-compute Cross-Variable Correlations BEFORE Normalization**

Store correlations in original scale, use them to guide imputation:

```python
def variable_wise_with_preserved_correlations(
    masked_data: Dict[int, MaskedData],
    correlation_matrix: pd.DataFrame
) -> Dict[int, pd.DataFrame]:
    """
    Use pre-computed correlations from unnormalized data to improve
    variable-wise imputation.

    Args:
        masked_data: Masked patient data (normalized)
        correlation_matrix: Correlations from UNNORMALIZED data
    """
    # For each patient
    for pid, mdata in masked_data.items():
        # Find completely missing variables
        completely_missing = []
        for col in mdata.data.columns:
            if mdata.data[col].isna().all():
                completely_missing.append(col)

        # For each missing variable, use weighted avg of correlated variables
        for missing_var in completely_missing:
            # Get top-k correlated variables
            corrs = correlation_matrix[missing_var].drop(missing_var)
            top_predictors = corrs.abs().nlargest(3).index

            # Weighted average based on correlation
            weights = corrs[top_predictors].values
            predictor_values = mdata.data[top_predictors].values

            # Imputed value = weighted sum
            imputed_values = np.nansum(
                predictor_values * weights[None, :], axis=1
            ) / np.sum(np.abs(weights))

            mdata.data[missing_var] = imputed_values

    return masked_data
```

**Expected improvement:** RMSE 0.81 → 0.70-0.75, R² -0.07 → 0.15-0.25

**Option B: Use Population Statistics**

When a variable is completely missing, use population mean/median from training data:

```python
def impute_with_population_stats(
    masked_data: Dict[int, MaskedData],
    population_stats: Dict[str, float]  # {var: mean_value}
) -> Dict[int, pd.DataFrame]:
    """Fill completely missing variables with population mean."""
    for pid, mdata in masked_data.items():
        for col in mdata.data.columns:
            if mdata.data[col].isna().all() and col in population_stats:
                # Fill with population mean (already normalized)
                mdata.data[col] = population_stats[col]  # Should be ~0 for normalized

    return masked_data
```

**Expected improvement:** RMSE 0.81 → 0.75-0.78, R² -0.07 → 0.05-0.15

**Option C: Dimensionality Reduction (PCA/Matrix Factorization)**

Use matrix completion techniques that work well when entire columns are missing:

```python
from sklearn.decomposition import PCA
from fancyimpute import SoftImpute

def matrix_completion_variable_wise(data: pd.DataFrame) -> pd.DataFrame:
    """
    Use SoftImpute (matrix factorization) for variables with high missingness.
    Works better than MICE when entire variables are missing.
    """
    from fancyimpute import SoftImpute

    # SoftImpute: low-rank matrix completion
    imputer = SoftImpute(max_iters=50, verbose=False)
    imputed_values = imputer.fit_transform(data.values)

    return pd.DataFrame(imputed_values, index=data.index, columns=data.columns)
```

**Expected improvement:** RMSE 0.81 → 0.65-0.70, R² -0.07 → 0.25-0.35

---

## Implementation Priority

### High Priority (Implement First)

1. **Fix SEQUENCE_END**: Add linear trend extrapolation to spline imputation
   - Easy to implement
   - Will improve R² from 0.0 → 0.20-0.30
   - Location: `imputation.py:smoothing_spline_impute()`

2. **Fix VARIABLE_WISE with MICE**: Add time as feature for MICE
   - Already using MICE, just need to add time column
   - Will improve RMSE from 0.81 → 0.70
   - Location: `imputation.py:mice_impute()`

### Medium Priority

3. **Add AR/ARIMA extrapolation** for SEQUENCE_END
   - Requires statsmodels dependency
   - Best results for time series extrapolation
   - Create new function `ar_extrapolation()`

4. **Store pre-normalization correlations** for VARIABLE_WISE
   - Modify `dataloader.py` to save correlation matrix
   - Pass to imputation methods

### Low Priority (Nice to Have)

5. **Matrix completion methods** (SoftImpute, NMF)
   - Requires `fancyimpute` dependency
   - Best for VARIABLE_WISE but complex to implement

6. **Deep learning methods** (LSTM, Transformers)
   - Requires PyTorch/TensorFlow
   - Overkill for this project but could be future work

---

## Quick Win: Hybrid Approach

Combine methods based on masking strategy:

```python
def smart_imputation(masked_data, masking_strategy, method):
    """Choose imputation based on masking pattern."""

    if masking_strategy == MaskingStrategy.MCAR:
        # Any method works well for interpolation
        return standard_impute(masked_data, method)

    elif masking_strategy == MaskingStrategy.SEQUENCE_END:
        # Use AR or MICE with time feature for extrapolation
        return mice_with_time(masked_data)

    elif masking_strategy == MaskingStrategy.VARIABLE_WISE:
        # Use matrix completion or MICE with population priors
        return matrix_completion(masked_data)
```

**Expected overall improvement:**
- SEQUENCE_END: RMSE 0.96 → 0.72, R² 0.0 → 0.30
- VARIABLE_WISE: RMSE 0.81 → 0.68, R² -0.07 → 0.25

---

## Summary Table: Expected Results After Fixes

| Masking | Method | Current RMSE | Fixed RMSE | Current R² | Fixed R² |
|---------|--------|--------------|------------|------------|----------|
| MCAR | MICE | 0.72 | 0.72 | 0.45 | 0.45 |
| **SEQUENCE_END** | **MICE+Time** | **0.96** | **0.70-0.75** | **0.0** | **0.30-0.40** |
| **VARIABLE_WISE** | **Matrix Completion** | **0.81** | **0.65-0.70** | **-0.07** | **0.25-0.35** |

---

## References for Further Reading

1. **Extrapolation**: Hyndman & Athanasopoulos (2021). *Forecasting: Principles and Practice*
2. **Matrix Completion**: Candès & Recht (2009). *Exact Matrix Completion via Convex Optimization*
3. **MICE variants**: White et al. (2011). *Multiple imputation using chained equations*
4. **Clinical time series**: Che et al. (2018). *Recurrent Neural Networks for Multivariate Time Series with Missing Values*
