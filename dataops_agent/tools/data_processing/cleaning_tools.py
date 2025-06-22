import pandas as pd
import numpy as np
import re
import warnings
from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from ..data_processing.processors import (
    TextProcessor,
    NumericProcessor,
    DateTimeProcessor,
    ContactProcessor,
    MissingDataHandler,
    DataCleaner
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import pyjanitor with proper error handling
try:
    import janitor as pj
    PYJANITOR_AVAILABLE = True
except ImportError:
    PYJANITOR_AVAILABLE = False

def clean_dataframe_comprehensive(df: pd.DataFrame, 
                                config: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary for cleaning options
    
    Returns:
        Tuple of cleaned DataFrame and cleaning report
    """
    if config is None:
        config = {
            'fix_encoding': True,
            'normalize_whitespace': True,
            'handle_duplicates': True,
            'standardize_columns': True,
            'parse_dates': True,
            'normalize_text': True,
            'handle_outliers': False,
            'impute_missing': True,
            'validate_emails': True,
            'parse_phones': False  # Requires country specification
        }
    
    cleaner = DataCleaner()
    df_clean = df.copy()
    
    # Standardize column names
    if config.get('standardize_columns', True):
        if PYJANITOR_AVAILABLE:
            df_clean = df_clean.clean_names()
            cleaner.log_operation("Standardized column names", "Used pyjanitor clean_names()")
        else:
            # Fallback column name cleaning
            df_clean.columns = [col.lower().strip().replace(' ', '_') for col in df_clean.columns]
            cleaner.log_operation("Standardized column names", "Manual cleaning")
    
    # Handle duplicates
    if config.get('handle_duplicates', True):
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_rows = initial_rows - len(df_clean)
        if removed_rows > 0:
            cleaner.log_operation("Removed duplicates", f"Removed {removed_rows} duplicate rows")
    
    # Process text columns
    text_columns = df_clean.select_dtypes(include=['object', 'string']).columns
    
    for col in text_columns:
        if config.get('fix_encoding', True):
            df_clean[col] = df_clean[col].apply(TextProcessor.fix_encoding)
        
        if config.get('normalize_whitespace', True):
            df_clean[col] = df_clean[col].apply(TextProcessor.clean_whitespace)
        
        if config.get('normalize_text', True):
            df_clean[col] = df_clean[col].apply(lambda x: TextProcessor.normalize_unicode(x, remove_accents=False))
        
        # Email validation
        if config.get('validate_emails', True) and 'email' in col.lower():
            df_clean[col] = df_clean[col].apply(ContactProcessor.clean_email)
            cleaner.log_operation(f"Validated emails in {col}", "Cleaned and validated email format")
    
    # Date parsing
    if config.get('parse_dates', True):
        date_columns = [col for col in df_clean.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_columns:
            try:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                cleaner.log_operation(f"Parsed dates in {col}", "Converted to datetime format")
            except Exception as e:
                logger.warning(f"Could not parse dates in {col}: {e}")
    
    # Handle numeric outliers
    if config.get('handle_outliers', False):
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df_clean[col] = NumericProcessor.handle_outliers(df_clean[col])
            cleaner.log_operation(f"Handled outliers in {col}", "Used IQR method")
    
    # Impute missing values
    if config.get('impute_missing', True):
        df_clean = MissingDataHandler.impute_missing_values(df_clean, strategy='mean')
        cleaner.log_operation("Imputed missing values", "Used mean/mode imputation")
    
    # Generate cleaning report
    cleaning_report = {
        'original_shape': df.shape,
        'cleaned_shape': df_clean.shape,
        'operations_log': cleaner.get_cleaning_report(),
        'missing_data_analysis': MissingDataHandler.analyze_missing_data(df_clean),
        'data_types': df_clean.dtypes.to_dict()
    }
    
    return df_clean, cleaning_report

def quick_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Quick data cleaning with sensible defaults."""
    cleaned_df, _ = clean_dataframe_comprehensive(df)
    return cleaned_df
