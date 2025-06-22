"""
Data Profiling and Analysis Tools

This module provides comprehensive data profiling capabilities including
statistical analysis, distribution analysis, and data quality assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import optional dependencies
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    from scipy.stats import kstest, jarque_bera, shapiro
    SCIPY_STATS_AVAILABLE = True
except ImportError:
    SCIPY_STATS_AVAILABLE = False

logger = logging.getLogger(__name__)


def profile_dataframe_comprehensive(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive profile of a DataFrame including statistical summaries,
    data types, missing values, duplicates, and distribution analysis.
    
    Args:
        df: Input DataFrame to profile
        
    Returns:
        Dictionary containing comprehensive profiling information
    """
    profile = {
        'basic_info': {
            'shape': df.shape,
            'columns': list(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'dtypes': df.dtypes.to_dict()
        },
        'missing_data': {
            'total_missing': df.isnull().sum().sum(),
            'missing_by_column': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'rows_with_missing': df.isnull().any(axis=1).sum()
        },
        'duplicates': {
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
        },
        'numeric_analysis': {},
        'categorical_analysis': {},
        'text_analysis': {},
        'correlation_analysis': {},
        'distribution_analysis': {}
    }
    
    # Analyze numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) > 0:
            profile['numeric_analysis'][col] = {
                'count': len(series),
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'median': series.median(),
                'q25': series.quantile(0.25),
                'q75': series.quantile(0.75),
                'skewness': series.skew(),
                'kurtosis': series.kurtosis(),
                'outliers_iqr': _detect_outliers_iqr(series),
                'unique_values': series.nunique(),
                'zeros': (series == 0).sum(),
                'negatives': (series < 0).sum()
            }
    
    # Analyze categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        series = df[col].dropna()
        if len(series) > 0:
            value_counts = series.value_counts()
            profile['categorical_analysis'][col] = {
                'count': len(series),
                'unique_values': series.nunique(),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
                'value_counts': value_counts.head(10).to_dict(),
                'entropy': _calculate_entropy(series)
            }
    
    # Text analysis for string columns
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        series = df[col].dropna().astype(str)
        if len(series) > 0:
            lengths = series.str.len()
            profile['text_analysis'][col] = {
                'avg_length': lengths.mean(),
                'min_length': lengths.min(),
                'max_length': lengths.max(),
                'std_length': lengths.std(),
                'empty_strings': (series == '').sum(),
                'whitespace_only': series.str.strip().eq('').sum(),
                'contains_numbers': series.str.contains(r'\d').sum(),
                'contains_special_chars': series.str.contains(r'[^a-zA-Z0-9\s]').sum()
            }
    
    # Correlation analysis
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        profile['correlation_analysis'] = {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlations': _find_high_correlations(corr_matrix, threshold=0.7)
        }
    
    return profile


def detect_data_types_advanced(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Advanced data type detection that goes beyond pandas' basic inference.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with suggested data types and confidence scores
    """
    suggestions = {}
    
    for col in df.columns:
        series = df[col].dropna()
        suggestions[col] = {
            'current_type': str(df[col].dtype),
            'suggested_type': None,
            'confidence': 0.0,
            'patterns': []
        }
        
        if len(series) == 0:
            continue
            
        # Convert to string for pattern matching
        str_series = series.astype(str)
        
        # Check for various patterns
        patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^[\+]?[1-9]?[0-9]{7,15}$',
            'url': r'^https?://[^\s/$.?#].[^\s]*$',
            'date': r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}',
            'time': r'\d{2}:\d{2}(:\d{2})?',
            'currency': r'[\$£€¥₹][0-9,.]+',
            'percentage': r'\d+\.?\d*%',
            'postal_code': r'^\d{5}(-\d{4})?$|^[A-Z]\d[A-Z] \d[A-Z]\d$',
            'ip_address': r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$',
            'boolean': r'^(true|false|yes|no|1|0)$'
        }
        
        detected_patterns = []
        for pattern_name, pattern in patterns.items():
            matches = str_series.str.match(pattern, case=False).sum()
            if matches > 0:
                confidence = matches / len(str_series)
                if confidence > 0.5:  # More than 50% match
                    detected_patterns.append({
                        'pattern': pattern_name,
                        'confidence': confidence,
                        'matches': matches
                    })
        
        suggestions[col]['patterns'] = detected_patterns
        
        # Suggest best type based on patterns and current type
        if detected_patterns:
            best_pattern = max(detected_patterns, key=lambda x: x['confidence'])
            suggestions[col]['suggested_type'] = best_pattern['pattern']
            suggestions[col]['confidence'] = best_pattern['confidence']
    
    return suggestions


def analyze_data_distributions(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Analyze the statistical distributions of numeric columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with distribution analysis for each numeric column
    """
    distributions = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 3:  # Need at least 3 values for meaningful analysis
            continue
            
        analysis = {
            'distribution_tests': {},
            'suggested_distribution': None,
            'normality_tests': {}
        }
        
        if SCIPY_STATS_AVAILABLE and len(series) >= 8:
            # Test for normality
            if len(series) <= 5000:  # Shapiro-Wilk for smaller samples
                try:
                    stat, p_value = shapiro(series)
                    analysis['normality_tests']['shapiro_wilk'] = {
                        'statistic': stat,
                        'p_value': p_value,
                        'is_normal': p_value > 0.05
                    }
                except Exception:
                    pass
            
            # Jarque-Bera test
            try:
                stat, p_value = jarque_bera(series)
                analysis['normality_tests']['jarque_bera'] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'is_normal': p_value > 0.05
                }
            except Exception:
                pass
            
            # Test against common distributions
            distributions_to_test = ['norm', 'expon', 'uniform', 'gamma']
            for dist_name in distributions_to_test:
                try:
                    dist = getattr(stats, dist_name)
                    params = dist.fit(series)
                    ks_stat, p_value = kstest(series, lambda x: dist.cdf(x, *params))
                    analysis['distribution_tests'][dist_name] = {
                        'ks_statistic': ks_stat,
                        'p_value': p_value,
                        'parameters': params
                    }
                except Exception:
                    continue
            
            # Suggest best fitting distribution
            if analysis['distribution_tests']:
                best_dist = max(analysis['distribution_tests'].items(), 
                              key=lambda x: x[1]['p_value'])
                analysis['suggested_distribution'] = best_dist[0]
        
        distributions[col] = analysis
    
    return distributions


def generate_data_quality_score(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate an overall data quality score based on various metrics.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with quality scores and breakdown
    """
    scores = {
        'overall_score': 0.0,
        'component_scores': {},
        'recommendations': []
    }
    
    # Completeness score (based on missing data)
    missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    completeness_score = max(0, 1 - missing_ratio)
    scores['component_scores']['completeness'] = completeness_score
    
    # Uniqueness score (based on duplicates)
    duplicate_ratio = df.duplicated().sum() / len(df)
    uniqueness_score = max(0, 1 - duplicate_ratio)
    scores['component_scores']['uniqueness'] = uniqueness_score
    
    # Consistency score (based on data type consistency)
    consistency_score = 1.0  # Start with perfect score
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check for mixed types in object columns
            sample = df[col].dropna().head(100)
            type_variety = len(set(type(x).__name__ for x in sample))
            if type_variety > 1:
                consistency_score -= 0.1
    
    consistency_score = max(0, consistency_score)
    scores['component_scores']['consistency'] = consistency_score
    
    # Validity score (based on data format patterns)
    validity_score = 1.0
    type_suggestions = detect_data_types_advanced(df)
    for col, suggestion in type_suggestions.items():
        if suggestion['patterns']:
            # If we detected patterns, check how well they match
            best_pattern = max(suggestion['patterns'], key=lambda x: x['confidence'])
            if best_pattern['confidence'] < 0.8:
                validity_score -= 0.05
    
    validity_score = max(0, validity_score)
    scores['component_scores']['validity'] = validity_score
    
    # Calculate overall score (weighted average)
    weights = {'completeness': 0.3, 'uniqueness': 0.25, 'consistency': 0.25, 'validity': 0.2}
    overall_score = sum(scores['component_scores'][component] * weight 
                       for component, weight in weights.items())
    scores['overall_score'] = overall_score
    
    # Generate recommendations
    if completeness_score < 0.8:
        scores['recommendations'].append("Address missing data issues")
    if uniqueness_score < 0.9:
        scores['recommendations'].append("Remove duplicate records")
    if consistency_score < 0.9:
        scores['recommendations'].append("Standardize data types and formats")
    if validity_score < 0.8:
        scores['recommendations'].append("Validate and correct data formats")
    
    return scores


def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, 
                      name1: str = "Dataset 1", name2: str = "Dataset 2") -> Dict[str, Any]:
    """
    Compare two DataFrames and identify differences in structure and content.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        name1: Name for first DataFrame
        name2: Name for second DataFrame
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {
        'structural_differences': {},
        'schema_differences': {},
        'statistical_differences': {},
        'content_differences': {}
    }
    
    # Structural comparison
    comparison['structural_differences'] = {
        'shape_comparison': {
            name1: df1.shape,
            name2: df2.shape,
            'rows_diff': df2.shape[0] - df1.shape[0],
            'cols_diff': df2.shape[1] - df1.shape[1]
        },
        'columns_comparison': {
            'common_columns': list(set(df1.columns) & set(df2.columns)),
            'only_in_' + name1: list(set(df1.columns) - set(df2.columns)),
            'only_in_' + name2: list(set(df2.columns) - set(df1.columns))
        }
    }
    
    # Schema comparison for common columns
    common_cols = set(df1.columns) & set(df2.columns)
    schema_diffs = {}
    for col in common_cols:
        if str(df1[col].dtype) != str(df2[col].dtype):
            schema_diffs[col] = {
                name1: str(df1[col].dtype),
                name2: str(df2[col].dtype)
            }
    comparison['schema_differences'] = schema_diffs
    
    # Statistical comparison for numeric columns
    numeric_cols = list(set(df1.select_dtypes(include=[np.number]).columns) & 
                       set(df2.select_dtypes(include=[np.number]).columns))
    
    stat_diffs = {}
    for col in numeric_cols:
        stats1 = df1[col].describe()
        stats2 = df2[col].describe()
        stat_diffs[col] = {
            name1: stats1.to_dict(),
            name2: stats2.to_dict(),
            'mean_diff': stats2['mean'] - stats1['mean'],
            'std_diff': stats2['std'] - stats1['std']
        }
    comparison['statistical_differences'] = stat_diffs
    
    return comparison


# Helper functions
def _detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> int:
    """Detect outliers using IQR method."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return ((series < lower_bound) | (series > upper_bound)).sum()


def _calculate_entropy(series: pd.Series) -> float:
    """Calculate Shannon entropy of a categorical series."""
    value_counts = series.value_counts()
    probabilities = value_counts / len(series)
    return -sum(probabilities * np.log2(probabilities))


def _find_high_correlations(corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict[str, Any]]:
    """Find pairs of variables with high correlation."""
    high_corrs = []
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:  # Avoid duplicates and self-correlation
                corr_val = corr_matrix.loc[col1, col2]
                if abs(corr_val) >= threshold:
                    high_corrs.append({
                        'variable1': col1,
                        'variable2': col2,
                        'correlation': corr_val
                    })
    return high_corrs
