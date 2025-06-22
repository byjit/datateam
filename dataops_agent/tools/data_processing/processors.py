# Available functionality:
# - TextProcessor: Fix encoding, normalize Unicode, clean whitespace
# - NumericProcessor: Handle outliers, normalize data, extract numbers
# - DateTimeProcessor: Parse flexible dates, standardize formats
# - ContactProcessor: Parse phone numbers, clean emails
# - MissingDataHandler: Analyze and impute missing data

import pandas as pd
import numpy as np
import re
import warnings
from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging


# Import ftfy with proper error handling
try:
    import ftfy
    FTFY_AVAILABLE = True
except ImportError:
    FTFY_AVAILABLE = False

# Import unidecode with proper error handling
try:
    from unidecode import unidecode
    UNIDECODE_AVAILABLE = True
except ImportError:
    UNIDECODE_AVAILABLE = False

# Import phonenumbers with proper error handling
try:
    import phonenumbers
    from phonenumbers import geocoder, carrier
    PHONENUMBERS_AVAILABLE = True
except ImportError:
    PHONENUMBERS_AVAILABLE = False

# Import dateutil with proper error handling
try:
    from dateutil import parser as date_parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False

# Import missingno with proper error handling
try:
    import missingno as msno
    MISSINGNO_AVAILABLE = True
except ImportError:
    MISSINGNO_AVAILABLE = False



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class DataCleaner:
    """Comprehensive data cleaning and preprocessing class."""
    
    def __init__(self):
        self.cleaning_log = []
        
    def log_operation(self, operation: str, details: str = ""):
        """Log cleaning operations for audit trail."""
        self.cleaning_log.append({
            'operation': operation,
            'details': details,
            'timestamp': pd.Timestamp.now()
        })
        logger.info(f"Applied: {operation} - {details}")
    
    def get_cleaning_report(self) -> pd.DataFrame:
        """Get a report of all cleaning operations performed."""
        return pd.DataFrame(self.cleaning_log)


class TextProcessor:
    """Advanced text cleaning and normalization functions."""
    
    @staticmethod
    def fix_encoding(text: str) -> str:
        """Fix broken Unicode encoding using ftfy."""
        if not FTFY_AVAILABLE:
            return text
        try:
            return ftfy.fix_text(text)
        except Exception as e:
            logger.warning(f"Error fixing encoding: {e}")
            return text
    
    @staticmethod
    def normalize_unicode(text: str, remove_accents: bool = False) -> str:
        """Normalize Unicode text and optionally remove accents."""
        if pd.isna(text):
            return text
            
        # Fix encoding first
        text = TextProcessor.fix_encoding(text)
        
        # Remove accents if requested
        if remove_accents and UNIDECODE_AVAILABLE:
            text = unidecode(text)
        
        return text
    
    @staticmethod
    def clean_whitespace(text: str) -> str:
        """Clean various types of whitespace characters."""
        if pd.isna(text):
            return text
            
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove zero-width characters
        text = re.sub(r'[\u200b-\u200f\ufeff]', '', text)
        
        return text
    
    @staticmethod
    def remove_html_tags(text: str) -> str:
        """Remove HTML tags from text."""
        if pd.isna(text):
            return text
        return re.sub(r'<[^>]+>', '', text)
    
    @staticmethod
    def standardize_case(text: str, case_type: str = 'lower') -> str:
        """Standardize text case."""
        if pd.isna(text):
            return text
            
        case_functions = {
            'lower': str.lower,
            'upper': str.upper,
            'title': str.title,
            'sentence': lambda x: x.capitalize()
        }
        
        if case_type in case_functions:
            return case_functions[case_type](text)
        return text


class NumericProcessor:
    """Numeric data cleaning and preprocessing functions."""
    
    @staticmethod
    def extract_numbers(text: str) -> Optional[float]:
        """Extract numeric values from text."""
        if pd.isna(text):
            return np.nan
            
        # Remove common currency symbols and separators
        cleaned = re.sub(r'[,$€£¥₹]', '', str(text))
        
        # Extract number (including decimals)
        match = re.search(r'-?\d+\.?\d*', cleaned)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return np.nan
        return np.nan
    
    @staticmethod
    def handle_outliers(data: pd.Series, method: str = 'iqr', factor: float = 1.5) -> pd.Series:
        """Handle outliers using various methods."""
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            return data.clip(lower=lower_bound, upper=upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            return data[z_scores < factor]
        
        return data
    
    @staticmethod
    def normalize_data(data: pd.Series, method: str = 'standard') -> pd.Series:
        """Normalize numeric data using various methods."""
        if method == 'standard':
            return (data - data.mean()) / data.std()
        elif method == 'minmax':
            return (data - data.min()) / (data.max() - data.min())
        elif method == 'robust':
            median = data.median()
            mad = np.median(np.abs(data - median))
            return (data - median) / mad
        return data


class DateTimeProcessor:
    """Date and time processing functions."""
    
    @staticmethod
    def parse_flexible_date(date_str: str) -> Optional[pd.Timestamp]:
        """Parse dates in various formats using dateutil."""
        if not DATEUTIL_AVAILABLE or pd.isna(date_str):
            return None
            
        try:
            return pd.Timestamp(date_parser.parse(str(date_str)))
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def standardize_date_format(date_series: pd.Series, target_format: str = '%Y-%m-%d') -> pd.Series:
        """Standardize date formats in a series."""
        parsed_dates = date_series.apply(DateTimeProcessor.parse_flexible_date)
        return parsed_dates.dt.strftime(target_format)
    
    @staticmethod
    def extract_date_components(date_series: pd.Series) -> pd.DataFrame:
        """Extract date components (year, month, day, etc.)."""
        parsed_dates = date_series.apply(DateTimeProcessor.parse_flexible_date)
        
        return pd.DataFrame({
            'year': parsed_dates.dt.year,
            'month': parsed_dates.dt.month,
            'day': parsed_dates.dt.day,
            'weekday': parsed_dates.dt.dayofweek,
            'quarter': parsed_dates.dt.quarter,
            'is_weekend': parsed_dates.dt.dayofweek >= 5
        })


class ContactProcessor:
    """Phone number and contact information processing."""
    
    @staticmethod
    def parse_phone_number(phone: str, country_code: str = 'US') -> Dict[str, Any]:
        """Parse and validate phone numbers."""
        if not PHONENUMBERS_AVAILABLE:
            return {'valid': False, 'formatted': phone, 'error': 'phonenumbers library not available'}
            
        try:
            parsed = phonenumbers.parse(str(phone), country_code)
            
            return {
                'valid': phonenumbers.is_valid_number(parsed),
                'formatted': phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL),
                'national': phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.NATIONAL),
                'country': geocoder.description_for_number(parsed, 'en'),
                'carrier': carrier.name_for_number(parsed, 'en'),
                'number_type': phonenumbers.number_type(parsed)
            }
        except Exception as e:
            return {'valid': False, 'formatted': phone, 'error': str(e)}
    
    @staticmethod
    def clean_email(email: str) -> str:
        """Clean and validate email addresses."""
        if pd.isna(email):
            return email
            
        email = str(email).lower().strip()
        
        # Basic email validation regex
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if re.match(email_pattern, email):
            return email
        return ''


class MissingDataHandler:
    """Handle missing data with various strategies."""
    
    @staticmethod
    def analyze_missing_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        missing_stats = {
            'total_missing': df.isnull().sum().sum(),
            'missing_by_column': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'rows_with_missing': df.isnull().any(axis=1).sum(),
            'complete_rows': len(df) - df.isnull().any(axis=1).sum()
        }
        return missing_stats
    
    @staticmethod
    def visualize_missing_data(df: pd.DataFrame, save_path: Optional[str] = None):
        """Visualize missing data patterns using missingno."""
        if not MISSINGNO_AVAILABLE:
            logger.warning("missingno not available for visualization")
            return
            
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        msno.matrix(df, ax=axes[0, 0])
        msno.bar(df, ax=axes[0, 1])
        msno.heatmap(df, ax=axes[1, 0])
        msno.dendrogram(df, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def impute_missing_values(df: pd.DataFrame, strategy: str = 'mean', columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Impute missing values using various strategies."""
        df_copy = df.copy()
        
        if columns is None:
            columns = df.columns
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if df[col].dtype in ['object', 'string']:
                # For categorical data
                if strategy == 'mode':
                    mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                    df_copy[col].fillna(mode_value, inplace=True)
                else:
                    df_copy[col].fillna('Unknown', inplace=True)
            else:
                # For numeric data
                if strategy == 'mean':
                    df_copy[col].fillna(df[col].mean(), inplace=True)
                elif strategy == 'median':
                    df_copy[col].fillna(df[col].median(), inplace=True)
                elif strategy == 'mode':
                    mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 0
                    df_copy[col].fillna(mode_value, inplace=True)
                elif strategy == 'forward_fill':
                    df_copy[col].fillna(method='ffill', inplace=True)
                elif strategy == 'backward_fill':
                    df_copy[col].fillna(method='bfill', inplace=True)
        
        return df_copy