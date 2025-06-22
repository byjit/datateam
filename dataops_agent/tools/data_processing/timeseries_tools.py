"""
Time Series Processing Tools

This module provides specialized tools for time series data processing,
analysis, forecasting, and feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import optional dependencies
try:
    from scipy import signal
    from scipy.stats import zscore
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


def detect_time_series_patterns(df: pd.DataFrame, 
                               datetime_column: str,
                               value_columns: List[str],
                               freq: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Detect patterns in time series data including seasonality, trends, and anomalies.
    
    Args:
        df: Input DataFrame with time series data
        datetime_column: Name of datetime column
        value_columns: List of value columns to analyze
        freq: Frequency of the time series (e.g., 'D', 'H', 'M')
        
    Returns:
        Dictionary with pattern analysis for each value column
    """
    if datetime_column not in df.columns:
        raise ValueError(f"Datetime column '{datetime_column}' not found")
    
    # Prepare the data
    df_ts = df.copy()
    df_ts[datetime_column] = pd.to_datetime(df_ts[datetime_column])
    df_ts = df_ts.sort_values(datetime_column).set_index(datetime_column)
    
    if freq:
        df_ts = df_ts.asfreq(freq)
    
    patterns = {}
    
    for col in value_columns:
        if col not in df_ts.columns:
            continue
        
        series = df_ts[col].dropna()
        if len(series) < 10:  # Need sufficient data
            continue
        
        col_patterns = {
            'trend': {},
            'seasonality': {},
            'anomalies': {},
            'statistics': {},
            'stationarity': {}
        }
        
        # Basic statistics
        col_patterns['statistics'] = {
            'mean': float(series.mean()),
            'std': float(series.std()),
            'min': float(series.min()),
            'max': float(series.max()),
            'autocorrelation_lag1': float(series.autocorr(lag=1)) if len(series) > 1 else 0,
            'coefficient_of_variation': float(series.std() / series.mean()) if series.mean() != 0 else np.inf
        }
        
        # Trend detection
        time_index = np.arange(len(series))
        trend_coef = np.polyfit(time_index, series.values, 1)[0]
        col_patterns['trend'] = {
            'slope': float(trend_coef),
            'direction': 'increasing' if trend_coef > 0.01 else 'decreasing' if trend_coef < -0.01 else 'stable',
            'strength': abs(float(trend_coef))
        }
        
        # Seasonality detection using FFT
        if SCIPY_AVAILABLE and len(series) >= 24:
            try:
                fft = np.fft.fft(series.values)
                frequencies = np.fft.fftfreq(len(series))
                
                # Find dominant frequencies
                power_spectrum = np.abs(fft) ** 2
                dominant_freq_idx = np.argsort(power_spectrum)[-5:]  # Top 5 frequencies
                
                seasonal_periods = []
                for idx in dominant_freq_idx:
                    if frequencies[idx] > 0:  # Positive frequencies only
                        period = 1 / frequencies[idx]
                        if 2 <= period <= len(series) / 2:  # Reasonable period range
                            seasonal_periods.append(float(period))
                
                col_patterns['seasonality'] = {
                    'detected_periods': seasonal_periods,
                    'has_seasonality': len(seasonal_periods) > 0,
                    'dominant_period': seasonal_periods[0] if seasonal_periods else None
                }
            except Exception as e:
                logger.warning(f"Error in seasonality detection for {col}: {e}")
                col_patterns['seasonality'] = {'has_seasonality': False}
        
        # Anomaly detection using z-score
        if len(series) > 3:
            z_scores = np.abs(zscore(series))
            anomaly_threshold = 3.0
            anomaly_indices = np.where(z_scores > anomaly_threshold)[0]
            
            col_patterns['anomalies'] = {
                'count': len(anomaly_indices),
                'percentage': float(len(anomaly_indices) / len(series) * 100),
                'anomaly_dates': [series.index[i].isoformat() for i in anomaly_indices[:10]],  # Top 10
                'anomaly_values': [float(series.iloc[i]) for i in anomaly_indices[:10]]
            }
        
        # Stationarity test (simple version)
        # Calculate rolling statistics
        window_size = min(12, len(series) // 4)
        if window_size >= 2:
            rolling_mean = series.rolling(window=window_size).mean()
            rolling_std = series.rolling(window=window_size).std()
            
            # Test if rolling statistics are relatively stable
            mean_stability = rolling_mean.std() / series.mean() if series.mean() != 0 else np.inf
            std_stability = rolling_std.std() / series.std() if series.std() != 0 else np.inf
            
            col_patterns['stationarity'] = {
                'is_stationary': mean_stability < 0.1 and std_stability < 0.1,
                'mean_stability': float(mean_stability),
                'std_stability': float(std_stability)
            }
        
        patterns[col] = col_patterns
    
    return patterns


def create_time_series_features(df: pd.DataFrame, 
                               datetime_column: str,
                               value_columns: List[str],
                               lag_periods: List[int] = [1, 7, 30],
                               rolling_windows: List[int] = [7, 30]) -> pd.DataFrame:
    """
    Create comprehensive time series features including lags, rolling statistics, and temporal features.
    
    Args:
        df: Input DataFrame with time series data
        datetime_column: Name of datetime column
        value_columns: List of value columns to create features for
        lag_periods: List of lag periods to create
        rolling_windows: List of rolling window sizes
        
    Returns:
        DataFrame with additional time series features
    """
    if datetime_column not in df.columns:
        raise ValueError(f"Datetime column '{datetime_column}' not found")
    
    df_result = df.copy()
    df_result[datetime_column] = pd.to_datetime(df_result[datetime_column])
    df_result = df_result.sort_values(datetime_column)
    
    for col in value_columns:
        if col not in df_result.columns:
            continue
        
        # Lag features
        for lag in lag_periods:
            df_result[f'{col}_lag_{lag}'] = df_result[col].shift(lag)
        
        # Rolling statistics
        for window in rolling_windows:
            df_result[f'{col}_rolling_mean_{window}'] = df_result[col].rolling(window=window).mean()
            df_result[f'{col}_rolling_std_{window}'] = df_result[col].rolling(window=window).std()
            df_result[f'{col}_rolling_min_{window}'] = df_result[col].rolling(window=window).min()
            df_result[f'{col}_rolling_max_{window}'] = df_result[col].rolling(window=window).max()
            df_result[f'{col}_rolling_median_{window}'] = df_result[col].rolling(window=window).median()
        
        # Difference features
        df_result[f'{col}_diff_1'] = df_result[col].diff(1)
        df_result[f'{col}_diff_7'] = df_result[col].diff(7)
        
        # Percentage change
        df_result[f'{col}_pct_change_1'] = df_result[col].pct_change(1)
        df_result[f'{col}_pct_change_7'] = df_result[col].pct_change(7)
        
        # Cumulative features
        df_result[f'{col}_cumsum'] = df_result[col].cumsum()
        df_result[f'{col}_cummax'] = df_result[col].cummax()
        df_result[f'{col}_cummin'] = df_result[col].cummin()
        
        # Expanding window features
        df_result[f'{col}_expanding_mean'] = df_result[col].expanding().mean()
        df_result[f'{col}_expanding_std'] = df_result[col].expanding().std()
    
    return df_result


def detect_change_points(series: pd.Series, 
                        method: str = 'variance',
                        min_segment_length: int = 10) -> List[int]:
    """
    Detect change points in a time series.
    
    Args:
        series: Input time series
        method: Change point detection method ('variance', 'mean')
        min_segment_length: Minimum length of segments
        
    Returns:
        List of change point indices
    """
    if len(series) < min_segment_length * 2:
        return []
    
    change_points = []
    
    if method == 'variance':
        # Variance-based change point detection
        for i in range(min_segment_length, len(series) - min_segment_length):
            left_segment = series.iloc[:i]
            right_segment = series.iloc[i:]
            
            left_var = left_segment.var()
            right_var = right_segment.var()
            
            # Calculate variance ratio
            if left_var > 0 and right_var > 0:
                var_ratio = max(left_var, right_var) / min(left_var, right_var)
                if var_ratio > 2.0:  # Significant variance change
                    change_points.append(i)
    
    elif method == 'mean':
        # Mean-based change point detection
        window_size = min_segment_length
        rolling_mean = series.rolling(window=window_size).mean()
        
        for i in range(window_size, len(series) - window_size):
            left_mean = rolling_mean.iloc[i-1]
            right_mean = rolling_mean.iloc[i+1]
            
            if abs(right_mean - left_mean) > 2 * series.std():
                change_points.append(i)
    
    # Remove nearby change points
    filtered_change_points = []
    for cp in change_points:
        if not filtered_change_points or cp - filtered_change_points[-1] >= min_segment_length:
            filtered_change_points.append(cp)
    
    return filtered_change_points


def decompose_time_series(df: pd.DataFrame, 
                         datetime_column: str,
                         value_column: str,
                         period: Optional[int] = None,
                         model: str = 'additive') -> Dict[str, pd.Series]:
    """
    Decompose time series into trend, seasonal, and residual components.
    
    Args:
        df: Input DataFrame
        datetime_column: Name of datetime column
        value_column: Name of value column
        period: Seasonal period (auto-detected if None)
        model: 'additive' or 'multiplicative'
        
    Returns:
        Dictionary with trend, seasonal, and residual components
    """
    if datetime_column not in df.columns or value_column not in df.columns:
        raise ValueError("Required columns not found")
    
    df_ts = df.copy()
    df_ts[datetime_column] = pd.to_datetime(df_ts[datetime_column])
    df_ts = df_ts.sort_values(datetime_column).set_index(datetime_column)
    series = df_ts[value_column].dropna()
    
    if len(series) < 24:  # Need sufficient data
        logger.warning("Insufficient data for decomposition")
        return {}
    
    # Auto-detect period if not provided
    if period is None:
        # Simple period detection based on data frequency
        time_diff = series.index[1] - series.index[0]
        if time_diff.total_seconds() <= 3600:  # Hourly or less
            period = 24  # Daily seasonality
        elif time_diff.days == 1:  # Daily
            period = 7   # Weekly seasonality
        else:
            period = 12  # Default monthly seasonality
    
    # Simple decomposition using moving averages
    # Trend component (using centered moving average)
    trend = series.rolling(window=period, center=True).mean()
    
    # Detrended series
    if model == 'additive':
        detrended = series - trend
    else:  # multiplicative
        detrended = series / trend
    
    # Seasonal component (average by period)
    seasonal_avg = {}
    for i in range(period):
        period_values = detrended.iloc[i::period].dropna()
        if len(period_values) > 0:
            seasonal_avg[i] = period_values.mean()
        else:
            seasonal_avg[i] = 0
    
    # Create seasonal series
    seasonal = pd.Series(index=series.index, dtype=float)
    for i, idx in enumerate(series.index):
        seasonal.loc[idx] = seasonal_avg[i % period]
    
    # Residual component
    if model == 'additive':
        residual = series - trend - seasonal
    else:  # multiplicative
        residual = series / (trend * seasonal)
    
    return {
        'original': series,
        'trend': trend,
        'seasonal': seasonal,
        'residual': residual
    }


def create_time_series_clusters(df: pd.DataFrame, 
                               datetime_column: str,
                               value_columns: List[str],
                               n_clusters: int = 3,
                               feature_type: str = 'statistical') -> Dict[str, Any]:
    """
    Cluster time series based on their characteristics.
    
    Args:
        df: Input DataFrame with multiple time series
        datetime_column: Name of datetime column
        value_columns: List of value columns (different time series)
        n_clusters: Number of clusters to create
        feature_type: Type of features to use ('statistical', 'shape', 'fourier')
        
    Returns:
        Dictionary with clustering results
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("sklearn not available for clustering")
        return {}
    
    if datetime_column not in df.columns:
        raise ValueError(f"Datetime column '{datetime_column}' not found")
    
    df_ts = df.copy()
    df_ts[datetime_column] = pd.to_datetime(df_ts[datetime_column])
    df_ts = df_ts.sort_values(datetime_column)
    
    # Extract features for each time series
    features = []
    valid_columns = []
    
    for col in value_columns:
        if col not in df_ts.columns:
            continue
        
        series = df_ts[col].dropna()
        if len(series) < 10:
            continue
        
        if feature_type == 'statistical':
            # Statistical features
            col_features = [
                series.mean(),
                series.std(),
                series.skew(),
                series.kurtosis(),
                series.min(),
                series.max(),
                series.autocorr(lag=1) if len(series) > 1 else 0,
                series.diff().abs().mean(),  # Average absolute difference
                (series > series.mean()).sum() / len(series)  # Fraction above mean
            ]
        
        elif feature_type == 'shape':
            # Shape-based features using first differences
            diffs = series.diff().dropna()
            if len(diffs) > 0:
                col_features = [
                    (diffs > 0).sum() / len(diffs),  # Fraction of increases
                    diffs.abs().mean(),  # Average absolute change
                    diffs.std(),  # Volatility
                    len([i for i in range(1, len(diffs)) if diffs.iloc[i] * diffs.iloc[i-1] < 0]) / len(diffs)  # Direction changes
                ]
            else:
                col_features = [0, 0, 0, 0]
        
        elif feature_type == 'fourier':
            # Fourier transform features
            if len(series) >= 8:
                fft = np.fft.fft(series.values)
                power_spectrum = np.abs(fft) ** 2
                
                # Use top frequency components as features
                n_features = min(8, len(power_spectrum) // 2)
                col_features = power_spectrum[:n_features].tolist()
            else:
                col_features = [0] * 8
        
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        features.append(col_features)
        valid_columns.append(col)
    
    if len(features) < 2:
        logger.warning("Insufficient time series for clustering")
        return {}
    
    # Standardize features
    features_array = np.array(features)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # Organize results
    results = {
        'cluster_assignments': dict(zip(valid_columns, cluster_labels.tolist())),
        'cluster_centers': kmeans.cluster_centers_.tolist(),
        'feature_names': [f'feature_{i}' for i in range(features_array.shape[1])],
        'n_clusters': n_clusters,
        'inertia': float(kmeans.inertia_)
    }
    
    # Add cluster summaries
    cluster_summaries = {}
    for cluster_id in range(n_clusters):
        cluster_series = [col for col, label in results['cluster_assignments'].items() if label == cluster_id]
        cluster_summaries[cluster_id] = {
            'series_count': len(cluster_series),
            'series_names': cluster_series
        }
    
    results['cluster_summaries'] = cluster_summaries
    
    return results


def fill_missing_time_series(df: pd.DataFrame, 
                           datetime_column: str,
                           value_columns: List[str],
                           method: str = 'interpolate',
                           freq: Optional[str] = None) -> pd.DataFrame:
    """
    Fill missing values in time series data using various methods.
    
    Args:
        df: Input DataFrame
        datetime_column: Name of datetime column
        value_columns: List of value columns
        method: Filling method ('interpolate', 'forward_fill', 'backward_fill', 'seasonal')
        freq: Target frequency for resampling
        
    Returns:
        DataFrame with filled missing values
    """
    if datetime_column not in df.columns:
        raise ValueError(f"Datetime column '{datetime_column}' not found")
    
    df_result = df.copy()
    df_result[datetime_column] = pd.to_datetime(df_result[datetime_column])
    df_result = df_result.sort_values(datetime_column).set_index(datetime_column)
    
    # Resample to regular frequency if specified
    if freq:
        df_result = df_result.asfreq(freq)
    
    for col in value_columns:
        if col not in df_result.columns:
            continue
        
        if method == 'interpolate':
            df_result[col] = df_result[col].interpolate(method='time')
        
        elif method == 'forward_fill':
            df_result[col] = df_result[col].fillna(method='ffill')
        
        elif method == 'backward_fill':
            df_result[col] = df_result[col].fillna(method='bfill')
        
        elif method == 'seasonal':
            # Seasonal filling using same period from previous cycles
            period = 7  # Default weekly period
            for i in range(len(df_result)):
                if pd.isna(df_result[col].iloc[i]):
                    # Look for same day of week in previous weeks
                    for lookback in [period, 2*period, 3*period]:
                        if i >= lookback:
                            fill_value = df_result[col].iloc[i - lookback]
                            if not pd.isna(fill_value):
                                df_result[col].iloc[i] = fill_value
                                break
        
        elif method == 'mean':
            # Fill with overall mean
            df_result[col] = df_result[col].fillna(df_result[col].mean())
    
    # Reset index to return datetime as column
    df_result = df_result.reset_index()
    
    return df_result


def calculate_time_series_similarity(series1: pd.Series, 
                                   series2: pd.Series,
                                   method: str = 'correlation') -> float:
    """
    Calculate similarity between two time series.
    
    Args:
        series1: First time series
        series2: Second time series
        method: Similarity method ('correlation', 'dtw', 'euclidean')
        
    Returns:
        Similarity score
    """
    # Align series by their common index
    common_index = series1.index.intersection(series2.index)
    if len(common_index) < 2:
        return 0.0
    
    s1_aligned = series1.loc[common_index].dropna()
    s2_aligned = series2.loc[common_index].dropna()
    
    # Further align by dropping NaN pairs
    valid_pairs = pd.DataFrame({'s1': s1_aligned, 's2': s2_aligned}).dropna()
    
    if len(valid_pairs) < 2:
        return 0.0
    
    s1_clean = valid_pairs['s1']
    s2_clean = valid_pairs['s2']
    
    if method == 'correlation':
        return float(s1_clean.corr(s2_clean))
    
    elif method == 'euclidean':
        # Normalized euclidean distance (converted to similarity)
        distance = np.sqrt(np.sum((s1_clean - s2_clean) ** 2))
        max_distance = np.sqrt(np.sum((s1_clean.max() - s1_clean.min()) ** 2))
        return 1 - (distance / max_distance) if max_distance > 0 else 1.0
    
    elif method == 'cosine':
        # Cosine similarity
        dot_product = np.dot(s1_clean, s2_clean)
        norm1 = np.linalg.norm(s1_clean)
        norm2 = np.linalg.norm(s2_clean)
        
        if norm1 > 0 and norm2 > 0:
            return float(dot_product / (norm1 * norm2))
        else:
            return 0.0
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def generate_time_series_forecast_features(df: pd.DataFrame,
                                         datetime_column: str,
                                         value_column: str,
                                         forecast_horizon: int = 30) -> pd.DataFrame:
    """
    Generate features specifically for time series forecasting.
    
    Args:
        df: Input DataFrame
        datetime_column: Name of datetime column
        value_column: Name of value column to forecast
        forecast_horizon: Number of periods to forecast
        
    Returns:
        DataFrame with forecasting features
    """
    if datetime_column not in df.columns or value_column not in df.columns:
        raise ValueError("Required columns not found")
    
    df_result = df.copy()
    df_result[datetime_column] = pd.to_datetime(df_result[datetime_column])
    df_result = df_result.sort_values(datetime_column)
    
    # Create basic time features
    df_result['year'] = df_result[datetime_column].dt.year
    df_result['month'] = df_result[datetime_column].dt.month
    df_result['day'] = df_result[datetime_column].dt.day
    df_result['dayofweek'] = df_result[datetime_column].dt.dayofweek
    df_result['quarter'] = df_result[datetime_column].dt.quarter
    df_result['is_weekend'] = df_result['dayofweek'] >= 5
    
    # Cyclical encoding for better ML model performance
    df_result['month_sin'] = np.sin(2 * np.pi * df_result['month'] / 12)
    df_result['month_cos'] = np.cos(2 * np.pi * df_result['month'] / 12)
    df_result['day_sin'] = np.sin(2 * np.pi * df_result['day'] / 31)
    df_result['day_cos'] = np.cos(2 * np.pi * df_result['day'] / 31)
    df_result['dayofweek_sin'] = np.sin(2 * np.pi * df_result['dayofweek'] / 7)
    df_result['dayofweek_cos'] = np.cos(2 * np.pi * df_result['dayofweek'] / 7)
    
    # Lag features based on forecast horizon
    lag_periods = [1, 7, 14, 30, 365] if forecast_horizon <= 30 else [1, 7, 30, 90, 365]
    for lag in lag_periods:
        df_result[f'{value_column}_lag_{lag}'] = df_result[value_column].shift(lag)
    
    # Rolling window features
    windows = [7, 14, 30, 90]
    for window in windows:
        df_result[f'{value_column}_rolling_mean_{window}'] = df_result[value_column].rolling(window).mean()
        df_result[f'{value_column}_rolling_std_{window}'] = df_result[value_column].rolling(window).std()
        df_result[f'{value_column}_rolling_min_{window}'] = df_result[value_column].rolling(window).min()
        df_result[f'{value_column}_rolling_max_{window}'] = df_result[value_column].rolling(window).max()
    
    # Exponentially weighted features
    for alpha in [0.1, 0.3, 0.5]:
        df_result[f'{value_column}_ewm_{alpha}'] = df_result[value_column].ewm(alpha=alpha).mean()
    
    # Difference and change features
    for period in [1, 7, 30]:
        df_result[f'{value_column}_diff_{period}'] = df_result[value_column].diff(period)
        df_result[f'{value_column}_pct_change_{period}'] = df_result[value_column].pct_change(period)
    
    return df_result
