"""
Data Generation and Augmentation Tools

This module provides capabilities for synthetic data generation,
data augmentation, and dataset expansion techniques.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import optional dependencies
try:
    from faker import Faker
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

logger = logging.getLogger(__name__)


def generate_synthetic_data(schema: Dict[str, Dict[str, Any]], 
                          n_rows: int = 1000,
                          seed: Optional[int] = None) -> pd.DataFrame:
    """
    Generate synthetic data based on a schema definition.
    
    Args:
        schema: Dictionary defining column schemas
                Format: {column_name: {'type': 'numeric'|'categorical'|'text'|'datetime', 
                                     'params': {...}}}
        n_rows: Number of rows to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic data
    """
    if seed:
        np.random.seed(seed)
        if FAKER_AVAILABLE:
            fake = Faker()
            fake.seed_instance(seed)
    else:
        if FAKER_AVAILABLE:
            fake = Faker()
    
    data = {}
    
    for column, config in schema.items():
        col_type = config.get('type', 'numeric')
        params = config.get('params', {})
        
        if col_type == 'numeric':
            if params.get('distribution') == 'normal':
                mean = params.get('mean', 0)
                std = params.get('std', 1)
                data[column] = np.random.normal(mean, std, n_rows)
            elif params.get('distribution') == 'uniform':
                low = params.get('low', 0)
                high = params.get('high', 1)
                data[column] = np.random.uniform(low, high, n_rows)
            elif params.get('distribution') == 'integer':
                low = params.get('low', 0)
                high = params.get('high', 100)
                data[column] = np.random.randint(low, high, n_rows)
            else:
                # Default to normal distribution
                data[column] = np.random.normal(0, 1, n_rows)
        
        elif col_type == 'categorical':
            categories = params.get('categories', ['A', 'B', 'C'])
            probabilities = params.get('probabilities', None)
            data[column] = np.random.choice(categories, n_rows, p=probabilities)
        
        elif col_type == 'boolean':
            prob_true = params.get('prob_true', 0.5)
            data[column] = np.random.choice([True, False], n_rows, p=[prob_true, 1-prob_true])
        
        elif col_type == 'datetime':
            start_date = params.get('start_date', '2020-01-01')
            end_date = params.get('end_date', '2023-12-31')
            
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            
            # Generate random dates
            date_range = pd.date_range(start_date, end_date, freq='D')
            data[column] = np.random.choice(date_range, n_rows)
        
        elif col_type == 'text' and FAKER_AVAILABLE:
            text_type = params.get('text_type', 'sentence')
            if text_type == 'name':
                data[column] = [fake.name() for _ in range(n_rows)]
            elif text_type == 'email':
                data[column] = [fake.email() for _ in range(n_rows)]
            elif text_type == 'address':
                data[column] = [fake.address() for _ in range(n_rows)]
            elif text_type == 'sentence':
                data[column] = [fake.sentence() for _ in range(n_rows)]
            elif text_type == 'paragraph':
                data[column] = [fake.paragraph() for _ in range(n_rows)]
            elif text_type == 'company':
                data[column] = [fake.company() for _ in range(n_rows)]
            elif text_type == 'phone':
                data[column] = [fake.phone_number() for _ in range(n_rows)]
            else:
                data[column] = [fake.sentence() for _ in range(n_rows)]
        
        elif col_type == 'text' and not FAKER_AVAILABLE:
            # Fallback text generation
            logger.warning("Faker not available, using simple text generation")
            data[column] = [f"text_{i}" for i in range(n_rows)]
        
        else:
            logger.warning(f"Unknown column type: {col_type}")
            data[column] = [None] * n_rows
    
    return pd.DataFrame(data)


def augment_data_with_noise(df: pd.DataFrame, 
                           noise_level: float = 0.1,
                           columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Augment numeric data by adding random noise.
    
    Args:
        df: Input DataFrame
        noise_level: Standard deviation of noise as fraction of column std
        columns: Specific columns to add noise to (default: all numeric)
        
    Returns:
        DataFrame with noisy data added
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    augmented_data = []
    
    for col in columns:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            col_std = df[col].std()
            noise = np.random.normal(0, noise_level * col_std, len(df))
            augmented_col = df[col] + noise
            augmented_data.append(augmented_col)
        else:
            augmented_data.append(df[col] if col in df.columns else None)
    
    # Create augmented DataFrame
    augmented_df = pd.DataFrame(dict(zip(columns, augmented_data)))
    
    # Add non-numeric columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        augmented_df[col] = df[col].values
    
    return augmented_df


def generate_smote_samples(df: pd.DataFrame, target_column: str, 
                          minority_class: Any = None,
                          n_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Generate synthetic samples using SMOTE-like technique for imbalanced datasets.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        minority_class: Value of minority class to oversample
        n_samples: Number of synthetic samples to generate
        
    Returns:
        DataFrame with synthetic samples
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("sklearn not available for SMOTE-like sampling")
        return df
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    
    # Get minority class if not specified
    if minority_class is None:
        class_counts = df[target_column].value_counts()
        minority_class = class_counts.idxmin()
    
    # Filter minority class samples
    minority_samples = df[df[target_column] == minority_class]
    
    if len(minority_samples) < 2:
        logger.warning("Not enough minority samples for SMOTE-like generation")
        return pd.DataFrame()
    
    # Select numeric features
    numeric_cols = minority_samples.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    if not numeric_cols:
        logger.warning("No numeric features found for SMOTE-like generation")
        return pd.DataFrame()
    
    X_minority = minority_samples[numeric_cols]
    
    # Fit k-nearest neighbors
    k = min(5, len(minority_samples) - 1)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_minority)
    
    # Generate synthetic samples
    if n_samples is None:
        n_samples = len(minority_samples)
    
    synthetic_samples = []
    
    for _ in range(n_samples):
        # Random minority sample
        sample_idx = np.random.randint(0, len(X_minority))
        sample = X_minority.iloc[sample_idx].values
        
        # Find k nearest neighbors
        _, indices = nn.kneighbors([sample])
        neighbor_idx = np.random.choice(indices[0][1:])  # Exclude the sample itself
        neighbor = X_minority.iloc[neighbor_idx].values
        
        # Generate synthetic sample
        alpha = np.random.random()
        synthetic_sample = sample + alpha * (neighbor - sample)
        synthetic_samples.append(synthetic_sample)
    
    # Create synthetic DataFrame
    synthetic_df = pd.DataFrame(synthetic_samples, columns=numeric_cols)
    synthetic_df[target_column] = minority_class
    
    # Add categorical columns with random sampling from minority class
    categorical_cols = minority_samples.select_dtypes(exclude=[np.number]).columns.tolist()
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)
    
    for col in categorical_cols:
        synthetic_df[col] = np.random.choice(minority_samples[col].values, n_samples)
    
    return synthetic_df


def create_time_series_variations(df: pd.DataFrame, 
                                 datetime_column: str,
                                 value_columns: List[str],
                                 variation_type: str = 'seasonal') -> pd.DataFrame:
    """
    Create time series variations for data augmentation.
    
    Args:
        df: Input DataFrame with time series data
        datetime_column: Name of datetime column
        value_columns: List of value columns to create variations for
        variation_type: Type of variation ('seasonal', 'trend', 'noise')
        
    Returns:
        DataFrame with time series variations
    """
    if datetime_column not in df.columns:
        raise ValueError(f"Datetime column '{datetime_column}' not found")
    
    df_copy = df.copy()
    
    # Ensure datetime column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df_copy[datetime_column]):
        df_copy[datetime_column] = pd.to_datetime(df_copy[datetime_column])
    
    # Sort by datetime
    df_copy = df_copy.sort_values(datetime_column)
    
    variations = []
    
    for col in value_columns:
        if col not in df_copy.columns or df_copy[col].dtype not in ['int64', 'float64']:
            continue
        
        if variation_type == 'seasonal':
            # Add seasonal component
            time_index = np.arange(len(df_copy))
            seasonal_component = np.sin(2 * np.pi * time_index / 365.25) * df_copy[col].std() * 0.1
            df_copy[f'{col}_seasonal'] = df_copy[col] + seasonal_component
        
        elif variation_type == 'trend':
            # Add linear trend
            time_index = np.arange(len(df_copy))
            trend_slope = df_copy[col].std() * 0.001  # Small trend
            trend_component = trend_slope * time_index
            df_copy[f'{col}_trend'] = df_copy[col] + trend_component
        
        elif variation_type == 'noise':
            # Add random noise
            noise = np.random.normal(0, df_copy[col].std() * 0.05, len(df_copy))
            df_copy[f'{col}_noisy'] = df_copy[col] + noise
        
        elif variation_type == 'lag':
            # Create lagged versions
            for lag in [1, 7, 30]:  # 1 day, 1 week, 1 month lags
                df_copy[f'{col}_lag_{lag}'] = df_copy[col].shift(lag)
        
        elif variation_type == 'rolling':
            # Create rolling statistics
            for window in [7, 30]:
                df_copy[f'{col}_rolling_mean_{window}'] = df_copy[col].rolling(window=window).mean()
                df_copy[f'{col}_rolling_std_{window}'] = df_copy[col].rolling(window=window).std()
    
    return df_copy


def expand_categorical_combinations(df: pd.DataFrame, 
                                  categorical_columns: List[str],
                                  max_combinations: int = 1000) -> pd.DataFrame:
    """
    Create new data by generating all possible combinations of categorical variables.
    
    Args:
        df: Input DataFrame
        categorical_columns: List of categorical columns
        max_combinations: Maximum number of combinations to generate
        
    Returns:
        DataFrame with all categorical combinations
    """
    # Get unique values for each categorical column
    unique_values = {}
    for col in categorical_columns:
        if col in df.columns:
            unique_values[col] = df[col].dropna().unique().tolist()
    
    if not unique_values:
        logger.warning("No valid categorical columns found")
        return pd.DataFrame()
    
    # Calculate total combinations
    total_combinations = 1
    for values in unique_values.values():
        total_combinations *= len(values)
    
    if total_combinations > max_combinations:
        logger.warning(f"Too many combinations ({total_combinations}), limiting to {max_combinations}")
    
    # Generate combinations
    from itertools import product
    
    combinations = list(product(*unique_values.values()))
    
    # Limit combinations if necessary
    if len(combinations) > max_combinations:
        combinations = combinations[:max_combinations]
    
    # Create DataFrame with combinations
    combination_df = pd.DataFrame(combinations, columns=categorical_columns)
    
    # Add default values for other columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in categorical_columns:
            combination_df[col] = df[col].mean()  # Use mean as default
    
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        if col not in categorical_columns:
            combination_df[col] = df[col].mode().iloc[0] if not df[col].mode().empty else 'default'
    
    return combination_df


def balance_dataset(df: pd.DataFrame, target_column: str, 
                   strategy: str = 'oversample',
                   random_state: Optional[int] = None) -> pd.DataFrame:
    """
    Balance a dataset by oversampling minority classes or undersampling majority classes.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        strategy: 'oversample', 'undersample', or 'smote'
        random_state: Random state for reproducibility
        
    Returns:
        Balanced DataFrame
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    
    if random_state:
        np.random.seed(random_state)
    
    class_counts = df[target_column].value_counts()
    
    if strategy == 'oversample':
        # Oversample minority classes to match majority class
        max_count = class_counts.max()
        balanced_dfs = []
        
        for class_value in class_counts.index:
            class_df = df[df[target_column] == class_value]
            current_count = len(class_df)
            
            if current_count < max_count:
                # Oversample by repeating samples
                n_repeats = max_count // current_count
                remainder = max_count % current_count
                
                oversampled_df = pd.concat([class_df] * n_repeats, ignore_index=True)
                
                if remainder > 0:
                    additional_samples = class_df.sample(n=remainder, replace=True, random_state=random_state)
                    oversampled_df = pd.concat([oversampled_df, additional_samples], ignore_index=True)
                
                balanced_dfs.append(oversampled_df)
            else:
                balanced_dfs.append(class_df)
        
        return pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=random_state)
    
    elif strategy == 'undersample':
        # Undersample majority classes to match minority class
        min_count = class_counts.min()
        balanced_dfs = []
        
        for class_value in class_counts.index:
            class_df = df[df[target_column] == class_value]
            
            if len(class_df) > min_count:
                undersampled_df = class_df.sample(n=min_count, random_state=random_state)
                balanced_dfs.append(undersampled_df)
            else:
                balanced_dfs.append(class_df)
        
        return pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=random_state)
    
    elif strategy == 'smote':
        # Use SMOTE-like technique for each minority class
        max_count = class_counts.max()
        balanced_dfs = [df]  # Start with original data
        
        for class_value in class_counts.index:
            current_count = class_counts[class_value]
            
            if current_count < max_count:
                n_synthetic = max_count - current_count
                synthetic_samples = generate_smote_samples(
                    df, target_column, class_value, n_synthetic
                )
                
                if not synthetic_samples.empty:
                    balanced_dfs.append(synthetic_samples)
        
        return pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=random_state)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def create_parallel_augmentation(df: pd.DataFrame, 
                                augmentation_functions: List[callable],
                                n_workers: int = 4) -> List[pd.DataFrame]:
    """
    Apply multiple augmentation functions in parallel using Dask.
    
    Args:
        df: Input DataFrame
        augmentation_functions: List of augmentation functions to apply
        n_workers: Number of parallel workers
        
    Returns:
        List of augmented DataFrames
    """
    if not DASK_AVAILABLE:
        logger.warning("Dask not available, running sequentially")
        return [func(df) for func in augmentation_functions]
    
    # Convert to Dask DataFrame
    ddf = dd.from_pandas(df, npartitions=n_workers)
    
    # Apply augmentation functions
    augmented_results = []
    
    for func in augmentation_functions:
        try:
            result = func(ddf).compute()
            augmented_results.append(result)
        except Exception as e:
            logger.error(f"Error applying augmentation function {func.__name__}: {e}")
            # Fallback to original function on pandas DataFrame
            try:
                result = func(df)
                augmented_results.append(result)
            except Exception as e2:
                logger.error(f"Fallback also failed for {func.__name__}: {e2}")
    
    return augmented_results


def infer_data_schema(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Infer a schema from an existing DataFrame that can be used for synthetic data generation.
    
    Args:
        df: Input DataFrame to analyze
        
    Returns:
        Schema dictionary that can be used with generate_synthetic_data
    """
    schema = {}
    
    for column in df.columns:
        col_info = {'type': 'numeric', 'params': {}}
        
        if df[column].dtype in ['int64', 'int32']:
            col_info['type'] = 'numeric'
            col_info['params'] = {
                'distribution': 'integer',
                'low': int(df[column].min()),
                'high': int(df[column].max())
            }
        
        elif df[column].dtype in ['float64', 'float32']:
            col_info['type'] = 'numeric'
            col_info['params'] = {
                'distribution': 'normal',
                'mean': float(df[column].mean()),
                'std': float(df[column].std())
            }
        
        elif df[column].dtype == 'bool':
            col_info['type'] = 'boolean'
            col_info['params'] = {
                'prob_true': float(df[column].mean())
            }
        
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            col_info['type'] = 'datetime'
            col_info['params'] = {
                'start_date': df[column].min(),
                'end_date': df[column].max()
            }
        
        elif df[column].dtype == 'object':
            unique_values = df[column].dropna().unique()
            
            if len(unique_values) <= 50:  # Treat as categorical
                col_info['type'] = 'categorical'
                value_counts = df[column].value_counts()
                probabilities = (value_counts / len(df)).tolist()
                col_info['params'] = {
                    'categories': unique_values.tolist(),
                    'probabilities': probabilities
                }
            else:  # Treat as text
                col_info['type'] = 'text'
                # Try to infer text type
                sample_values = df[column].dropna().astype(str).head(10)
                
                if sample_values.str.contains('@').any():
                    col_info['params']['text_type'] = 'email'
                elif sample_values.str.len().mean() > 50:
                    col_info['params']['text_type'] = 'paragraph'
                else:
                    col_info['params']['text_type'] = 'sentence'
        
        schema[column] = col_info
    
    return schema
