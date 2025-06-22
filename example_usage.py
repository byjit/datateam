"""
Example usage of the enhanced data processing tools

This script demonstrates the comprehensive data processing capabilities
of the DataOps Agent, including all new functions for profiling,
feature engineering, data generation, and time series analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
# Also import functions directly for easier use
from dataops_agent.tools.data_processing.profiling_tools import profile_dataframe_comprehensive
from dataops_agent.tools.data_processing.feature_engineering_tools import create_polynomial_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_data_profiling():
    """Demonstrate comprehensive data profiling capabilities."""
    print("\n=== DATA PROFILING DEMO ===")
    
    # Create sample data with various data quality issues
    np.random.seed(42)
    data = {
        'id': range(1000),
        'name': [f'User_{i}' if i % 10 != 0 else None for i in range(1000)],  # Some missing
        'email': [f'user{i}@email.com' if i % 15 != 0 else f'invalid_email_{i}' for i in range(1000)],
        'age': np.random.normal(35, 10, 1000).astype(int),
        'salary': np.random.lognormal(10, 0.5, 1000),
        'department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing'], 1000),
        'join_date': pd.date_range('2020-01-01', periods=1000, freq='D'),
        'is_active': np.random.choice([True, False], 1000, p=[0.8, 0.2])
    }
    
    # Add some duplicates and outliers
    df = pd.DataFrame(data)
    df.loc[50:52, :] = df.loc[0:2, :].values  # Add duplicates
    df.loc[100:105, 'salary'] = df.loc[100:105, 'salary'] * 10  # Add outliers
    
    print(f"Sample dataset shape: {df.shape}")
    
    # Comprehensive profiling
    profile = profile_dataframe_comprehensive(df)
    
    print("\nBasic Info:")
    for key, value in profile['basic_info'].items():
        print(f"  {key}: {value}")
    
    print("\nMissing Data Summary:")
    print(f"  Total missing values: {profile['missing_data']['total_missing']}")
    print(f"  Rows with missing data: {profile['missing_data']['rows_with_missing']}")
    
    print("\nDuplicate Analysis:")
    print(f"  Duplicate rows: {profile['duplicates']['duplicate_rows']}")
    print(f"  Duplicate percentage: {profile['duplicates']['duplicate_percentage']:.2f}%")
    
    print("\nNumeric Column Analysis (sample):")
    for col, stats in list(profile['numeric_analysis'].items())[:2]:
        print(f"  {col}:")
        print(f"    Mean: {stats['mean']:.2f}")
        print(f"    Std: {stats['std']:.2f}")
        print(f"    Outliers (IQR): {stats['outliers_iqr']}")
    
    return df


def demo_feature_engineering(df):
    """Demonstrate feature engineering capabilities."""
    print("\n=== FEATURE ENGINEERING DEMO ===")
    
    # Create polynomial features
    print("\n1. Creating polynomial features...")
    poly_df = create_polynomial_features(df, ['age', 'salary'], degree=2, interaction_only=True)
    print(f"  Original columns: {len(df.columns)}")
    print(f"  After polynomial features: {len(poly_df.columns)}")
    print(f"  New polynomial columns: {[col for col in poly_df.columns if col.startswith('poly_')][:3]}...")
    
    return poly_df



def demo_data_quality_assessment(df):
    """Demonstrate data quality assessment."""
    print("\n=== DATA QUALITY ASSESSMENT DEMO ===")
    
    from dataops_agent.tools.data_processing.profiling_tools import generate_data_quality_score
    from dataops_agent.tools.data_processing.validation_tools import validate_data_quality
    
    # Generate quality score
    quality_score = generate_data_quality_score(df)
    
    print(f"\nOverall Data Quality Score: {quality_score['overall_score']:.3f}")
    print("\nComponent Scores:")
    for component, score in quality_score['component_scores'].items():
        print(f"  {component.capitalize()}: {score:.3f}")
    
    print("\nRecommendations:")
    for rec in quality_score['recommendations']:
        print(f"  - {rec}")
    
    # Detailed validation
    validation_report = validate_data_quality(df)
    print(f"\nDetailed Validation:")
    print(f"  Shape: {validation_report['shape']}")
    print(f"  Memory usage: {validation_report['memory_usage']:.2f} MB")
    print(f"  Data types: {validation_report['data_types']}")


def main():
    """Run all demos."""
    print("🚀 DataOps Agent - Enhanced Data Processing Tools Demo")
    print("=" * 60)
    
    # Demo 1: Data Profiling
    sample_df = demo_data_profiling()
    
    # Demo 2: Feature Engineering
    engineered_df = demo_feature_engineering(sample_df)
    
    # Demo 5: Data Quality Assessment
    demo_data_quality_assessment(sample_df)
    
    print("\n" + "=" * 60)
    print("✅ Demo completed successfully!")
    print("\nThe DataOps Agent now includes:")
    print("  📊 Comprehensive data profiling and analysis")
    print("  🔧 Advanced feature engineering capabilities")
    print("  🎲 Synthetic data generation tools")
    print("  📈 Specialized time series processing")
    print("  ✅ Enhanced data quality assessment")
    print("  🛠️ All functions available as Google ADK tools")
    
    print(f"\nTotal available tools: 38")
    print("See documentation for complete function descriptions.")


if __name__ == "__main__":
    main()
