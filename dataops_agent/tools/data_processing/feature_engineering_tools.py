"""
Feature Engineering Tools

This module provides comprehensive feature engineering capabilities including
feature creation, selection, transformation, and encoding techniques.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Import optional dependencies
try:
    from sklearn.preprocessing import (
        StandardScaler, MinMaxScaler, RobustScaler, 
        LabelEncoder, OneHotEncoder, OrdinalEncoder
    )
    from sklearn.feature_selection import (
        SelectKBest, f_classif, f_regression, chi2,
        RFE, SelectFromModel
    )
    SKLEARN_PREPROCESSING_AVAILABLE = True
except ImportError:
    SKLEARN_PREPROCESSING_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    SKLEARN_ENSEMBLE_AVAILABLE = True
except ImportError:
    SKLEARN_ENSEMBLE_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_DECOMPOSITION_AVAILABLE = True
except ImportError:
    SKLEARN_DECOMPOSITION_AVAILABLE = False

SKLEARN_AVAILABLE = (SKLEARN_PREPROCESSING_AVAILABLE and 
                    SKLEARN_ENSEMBLE_AVAILABLE and 
                    SKLEARN_DECOMPOSITION_AVAILABLE)

logger = logging.getLogger(__name__)


def create_polynomial_features(df: pd.DataFrame, columns: List[str], 
                             degree: int = 2, interaction_only: bool = False) -> pd.DataFrame:
    """
    Create polynomial features from specified numeric columns.
    
    Args:
        df: Input DataFrame
        columns: List of column names to create polynomial features from
        degree: Degree of polynomial features
        interaction_only: If True, only create interaction features
        
    Returns:
        DataFrame with polynomial features added
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("sklearn not available for polynomial features")
        return df
    
    from sklearn.preprocessing import PolynomialFeatures
    
    df_result = df.copy()
    
    # Select only numeric columns that exist
    valid_columns = [col for col in columns if col in df.columns and 
                    df[col].dtype in ['int64', 'float64']]
    
    if not valid_columns:
        logger.warning("No valid numeric columns found for polynomial features")
        return df_result
    
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, 
                             include_bias=False)
    
    # Generate polynomial features
    poly_features = poly.fit_transform(df[valid_columns])
    
    # Create feature names
    feature_names = poly.get_feature_names_out(valid_columns)
    
    # Add new features to DataFrame
    for i, name in enumerate(feature_names):
        if name not in valid_columns:  # Don't duplicate original features
            df_result[f'poly_{name}'] = poly_features[:, i]
    
    return df_result


def create_binning_features(df: pd.DataFrame, column: str, 
                          bins: Union[int, List[float]], labels: Optional[List[str]] = None) -> pd.Series:
    """
    Create binned categorical features from continuous variables.
    
    Args:
        df: Input DataFrame
        column: Column name to bin
        bins: Number of bins or list of bin edges
        labels: Optional labels for bins
        
    Returns:
        Series with binned values
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    try:
        return pd.cut(df[column], bins=bins, labels=labels, include_lowest=True)
    except Exception as e:
        logger.error(f"Error creating bins for column '{column}': {e}")
        return df[column]


def create_aggregation_features(df: pd.DataFrame, group_columns: List[str], 
                               agg_columns: List[str], 
                               agg_functions: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
    """
    Create aggregation features based on grouping columns.
    
    Args:
        df: Input DataFrame
        group_columns: Columns to group by
        agg_columns: Columns to aggregate
        agg_functions: Aggregation functions to apply
        
    Returns:
        DataFrame with aggregation features added
    """
    df_result = df.copy()
    
    # Validate columns exist
    valid_group_cols = [col for col in group_columns if col in df.columns]
    valid_agg_cols = [col for col in agg_columns if col in df.columns and 
                     df[col].dtype in ['int64', 'float64']]
    
    if not valid_group_cols or not valid_agg_cols:
        logger.warning("No valid columns found for aggregation")
        return df_result
    
    try:
        # Create aggregations
        for agg_col in valid_agg_cols:
            for func in agg_functions:
                agg_name = f'{agg_col}_{func}_by_{"_".join(valid_group_cols)}'
                aggregated = df.groupby(valid_group_cols)[agg_col].transform(func)
                df_result[agg_name] = aggregated
    
    except Exception as e:
        logger.error(f"Error creating aggregation features: {e}")
    
    return df_result


def create_datetime_features(df: pd.DataFrame, datetime_columns: List[str]) -> pd.DataFrame:
    """
    Extract comprehensive datetime features from datetime columns.
    
    Args:
        df: Input DataFrame
        datetime_columns: List of datetime column names
        
    Returns:
        DataFrame with datetime features added
    """
    df_result = df.copy()
    
    for col in datetime_columns:
        if col not in df.columns:
            continue
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            try:
                df_result[col] = pd.to_datetime(df[col])
            except Exception:
                logger.warning(f"Could not convert column '{col}' to datetime")
                continue
        
        # Extract various datetime components
        dt_col = df_result[col]
        
        # Basic components
        df_result[f'{col}_year'] = dt_col.dt.year
        df_result[f'{col}_month'] = dt_col.dt.month
        df_result[f'{col}_day'] = dt_col.dt.day
        df_result[f'{col}_dayofweek'] = dt_col.dt.dayofweek
        df_result[f'{col}_hour'] = dt_col.dt.hour
        df_result[f'{col}_minute'] = dt_col.dt.minute
        
        # Advanced components
        df_result[f'{col}_quarter'] = dt_col.dt.quarter
        df_result[f'{col}_weekofyear'] = dt_col.dt.isocalendar().week
        df_result[f'{col}_dayofyear'] = dt_col.dt.dayofyear
        
        # Boolean features
        df_result[f'{col}_is_weekend'] = dt_col.dt.dayofweek >= 5
        df_result[f'{col}_is_month_start'] = dt_col.dt.is_month_start
        df_result[f'{col}_is_month_end'] = dt_col.dt.is_month_end
        df_result[f'{col}_is_quarter_start'] = dt_col.dt.is_quarter_start
        df_result[f'{col}_is_quarter_end'] = dt_col.dt.is_quarter_end
        
        # Cyclical features (sine/cosine encoding)
        df_result[f'{col}_month_sin'] = np.sin(2 * np.pi * dt_col.dt.month / 12)
        df_result[f'{col}_month_cos'] = np.cos(2 * np.pi * dt_col.dt.month / 12)
        df_result[f'{col}_hour_sin'] = np.sin(2 * np.pi * dt_col.dt.hour / 24)
        df_result[f'{col}_hour_cos'] = np.cos(2 * np.pi * dt_col.dt.hour / 24)
        df_result[f'{col}_dayofweek_sin'] = np.sin(2 * np.pi * dt_col.dt.dayofweek / 7)
        df_result[f'{col}_dayofweek_cos'] = np.cos(2 * np.pi * dt_col.dt.dayofweek / 7)
    
    return df_result


def create_text_features(df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
    """
    Extract features from text columns.
    
    Args:
        df: Input DataFrame
        text_columns: List of text column names
        
    Returns:
        DataFrame with text features added
    """
    df_result = df.copy()
    
    for col in text_columns:
        if col not in df.columns:
            continue
        
        text_series = df[col].fillna('').astype(str)
        
        # Basic text features
        df_result[f'{col}_length'] = text_series.str.len()
        df_result[f'{col}_word_count'] = text_series.str.split().str.len()
        df_result[f'{col}_char_count'] = text_series.str.len()
        df_result[f'{col}_sentence_count'] = text_series.str.count(r'[.!?]+')
        
        # Advanced text features
        df_result[f'{col}_avg_word_length'] = (
            text_series.str.replace(r'[^\w\s]', '', regex=True)
            .str.split()
            .apply(lambda x: np.mean([len(word) for word in x]) if x else 0)
        )
        
        # Character type features
        df_result[f'{col}_uppercase_count'] = text_series.str.count(r'[A-Z]')
        df_result[f'{col}_lowercase_count'] = text_series.str.count(r'[a-z]')
        df_result[f'{col}_digit_count'] = text_series.str.count(r'\d')
        df_result[f'{col}_special_char_count'] = text_series.str.count(r'[^\w\s]')
        df_result[f'{col}_whitespace_count'] = text_series.str.count(r'\s')
        
        # Boolean features
        df_result[f'{col}_has_url'] = text_series.str.contains(r'http[s]?://', case=False, na=False)
        df_result[f'{col}_has_email'] = text_series.str.contains(r'\S+@\S+', case=False, na=False)
        df_result[f'{col}_has_phone'] = text_series.str.contains(r'\d{3}-?\d{3}-?\d{4}', case=False, na=False)
        df_result[f'{col}_is_numeric'] = text_series.str.isnumeric()
        df_result[f'{col}_is_alpha'] = text_series.str.isalpha()
        df_result[f'{col}_is_alnum'] = text_series.str.isalnum()
    
    return df_result


def encode_categorical_features(df: pd.DataFrame, 
                              encoding_config: Dict[str, Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply various encoding techniques to categorical features.
    
    Args:
        df: Input DataFrame
        encoding_config: Configuration for encoding each column
                        Format: {column_name: {'method': 'onehot'|'label'|'ordinal'|'target', 
                                             'params': {...}}}
    
    Returns:
        Tuple of encoded DataFrame and encoding metadata
    """
    if not SKLEARN_PREPROCESSING_AVAILABLE:
        logger.warning("sklearn preprocessing not available for categorical encoding")
        return df, {}
    
    df_result = df.copy()
    encoding_metadata = {}
    
    for column, config in encoding_config.items():
        if column not in df.columns:
            continue
        
        method = config.get('method', 'onehot')
        params = config.get('params', {})
        
        try:
            if method == 'onehot':
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', **params)
                encoded = encoder.fit_transform(df[[column]])
                
                # Create feature names
                feature_names = [f'{column}_{cat}' for cat in encoder.categories_[0]]
                
                # Add encoded features
                for i, name in enumerate(feature_names):
                    df_result[name] = encoded[:, i]
                
                # Remove original column
                df_result = df_result.drop(columns=[column])
                
                encoding_metadata[column] = {
                    'method': 'onehot',
                    'encoder': encoder,
                    'feature_names': feature_names
                }
            
            elif method == 'label':
                encoder = LabelEncoder()
                df_result[f'{column}_encoded'] = encoder.fit_transform(df[column].fillna('Unknown'))
                
                encoding_metadata[column] = {
                    'method': 'label',
                    'encoder': encoder,
                    'classes': encoder.classes_.tolist()
                }
            
            elif method == 'ordinal':
                categories = params.get('categories', 'auto')
                encoder = OrdinalEncoder(categories=categories, handle_unknown='use_encoded_value',
                                       unknown_value=-1)
                df_result[f'{column}_ordinal'] = encoder.fit_transform(df[[column]]).flatten()
                
                encoding_metadata[column] = {
                    'method': 'ordinal',
                    'encoder': encoder,
                    'categories': encoder.categories_[0].tolist()
                }
            
            elif method == 'frequency':
                # Frequency encoding
                freq_map = df[column].value_counts().to_dict()
                df_result[f'{column}_frequency'] = df[column].map(freq_map)
                
                encoding_metadata[column] = {
                    'method': 'frequency',
                    'frequency_map': freq_map
                }
        
        except Exception as e:
            logger.error(f"Error encoding column '{column}' with method '{method}': {e}")
    
    return df_result, encoding_metadata


def select_features_statistical(df: pd.DataFrame, target_column: str, 
                               task_type: str = 'classification', 
                               k: int = 10) -> Tuple[List[str], Dict[str, float]]:
    """
    Select features using statistical tests.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        task_type: 'classification' or 'regression'
        k: Number of features to select
        
    Returns:
        Tuple of selected feature names and their scores
    """
    if not SKLEARN_PREPROCESSING_AVAILABLE:
        logger.warning("sklearn not available for feature selection")
        return list(df.columns), {}
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    
    # Prepare features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Select only numeric features for statistical tests
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X[numeric_features]
    
    if len(numeric_features) == 0:
        logger.warning("No numeric features found for statistical selection")
        return [], {}
    
    # Choose appropriate statistical test
    if task_type == 'classification':
        score_func = f_classif
    else:
        score_func = f_regression
    
    # Perform feature selection
    selector = SelectKBest(score_func=score_func, k=min(k, len(numeric_features)))
    X_selected = selector.fit_transform(X_numeric, y)
    
    # Get selected features and scores
    selected_features = X_numeric.columns[selector.get_support()].tolist()
    feature_scores = dict(zip(X_numeric.columns, selector.scores_))
    
    return selected_features, feature_scores


def select_features_model_based(df: pd.DataFrame, target_column: str, 
                               task_type: str = 'classification',
                               max_features: Optional[int] = None) -> Tuple[List[str], Dict[str, float]]:
    """
    Select features using a machine learning model.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        task_type: 'classification' or 'regression'
        max_features: Maximum number of features to select
        
    Returns:
        Tuple of selected feature names and their importance scores
    """
    if not SKLEARN_ENSEMBLE_AVAILABLE:
        logger.warning("sklearn ensemble models not available for model-based feature selection")
        return list(df.columns), {}
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    
    # Prepare features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Select only numeric features
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X[numeric_features]
    
    if len(numeric_features) == 0:
        logger.warning("No numeric features found for model-based selection")
        return [], {}
    
    # Choose appropriate model
    if task_type == 'classification':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Fit model and get feature importances
    model.fit(X_numeric, y)
    feature_importances = dict(zip(numeric_features, model.feature_importances_))
    
    # Select features based on importance
    if max_features:
        sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feat for feat, _ in sorted_features[:max_features]]
    else:
        # Use SelectFromModel with default threshold
        if SKLEARN_PREPROCESSING_AVAILABLE:
            selector = SelectFromModel(model)
            selector.fit(X_numeric, y)
            selected_features = X_numeric.columns[selector.get_support()].tolist()
        else:
            # Fallback: select top 10 features
            sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
            selected_features = [feat for feat, _ in sorted_features[:10]]
    
    return selected_features, feature_importances


def apply_dimensionality_reduction(df: pd.DataFrame, 
                                 method: str = 'pca', 
                                 n_components: Optional[int] = None,
                                 **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply dimensionality reduction techniques.
    
    Args:
        df: Input DataFrame (should contain only numeric columns)
        method: 'pca' or 'tsne'
        n_components: Number of components to keep
        **kwargs: Additional parameters for the method
        
    Returns:
        Tuple of transformed DataFrame and metadata
    """
    if not SKLEARN_DECOMPOSITION_AVAILABLE:
        logger.warning("sklearn decomposition not available for dimensionality reduction")
        return df, {}
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        logger.warning("No numeric columns found for dimensionality reduction")
        return df, {}
    
    metadata = {'method': method, 'original_shape': numeric_df.shape}
    
    if method == 'pca':
        reducer = PCA(n_components=n_components, **kwargs)
        transformed = reducer.fit_transform(numeric_df)
        
        # Create component names
        component_names = [f'PC{i+1}' for i in range(transformed.shape[1])]
        
        metadata.update({
            'explained_variance_ratio': reducer.explained_variance_ratio_.tolist(),
            'cumulative_variance_ratio': np.cumsum(reducer.explained_variance_ratio_).tolist(),
            'components': reducer.components_.tolist(),
            'n_components': reducer.n_components_
        })
    
    elif method == 'tsne':
        if n_components is None:
            n_components = 2
        
        reducer = TSNE(n_components=n_components, **kwargs)
        transformed = reducer.fit_transform(numeric_df)
        
        component_names = [f'TSNE{i+1}' for i in range(transformed.shape[1])]
        
        metadata.update({
            'n_components': n_components,
            'kl_divergence': reducer.kl_divergence_
        })
    
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    # Create result DataFrame
    result_df = pd.DataFrame(transformed, columns=component_names, index=df.index)
    
    # Add non-numeric columns back
    non_numeric_df = df.select_dtypes(exclude=[np.number])
    if not non_numeric_df.empty:
        result_df = pd.concat([result_df, non_numeric_df], axis=1)
    
    metadata['new_shape'] = result_df.shape
    
    return result_df, metadata
