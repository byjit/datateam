import pandas as pd
from typing import Dict, Any
from ..data_processing.processors import MissingDataHandler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate data quality and provide recommendations.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Dictionary with quality metrics and recommendations
    """
    quality_report = {
        'shape': df.shape,
        'missing_data': MissingDataHandler.analyze_missing_data(df),
        'data_types': df.dtypes.value_counts().to_dict(),
        'duplicates': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        'recommendations': []
    }
    
    # Generate recommendations
    if quality_report['missing_data']['total_missing'] > 0:
        quality_report['recommendations'].append(
            f"Address {quality_report['missing_data']['total_missing']} missing values"
        )
    
    if quality_report['duplicates'] > 0:
        quality_report['recommendations'].append(
            f"Remove {quality_report['duplicates']} duplicate rows"
        )
    
    # Check for potential issues in text columns
    text_columns = df.select_dtypes(include=['object', 'string']).columns
    for col in text_columns:
        sample_values = df[col].dropna().head(100)
        
        # Check for encoding issues
        if any('â€' in str(val) for val in sample_values):
            quality_report['recommendations'].append(f"Fix encoding issues in column '{col}'")
        
        # Check for excessive whitespace
        if any(len(str(val)) != len(str(val).strip()) for val in sample_values):
            quality_report['recommendations'].append(f"Clean whitespace in column '{col}'")
    
    return quality_report
