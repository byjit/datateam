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

# Import all the new data processing tools
from dataops_agent.tools.data_processing.data_processing_tools import (
    # Core tools
    clean_dataframe_comprehensive_tool,
    validate_data_quality_tool,
    
    # Profiling tools
    profile_dataframe_comprehensive_tool,
    detect_data_types_advanced_tool,
    generate_data_quality_score_tool,
    
    # Feature engineering tools
    create_polynomial_features_tool,
    create_datetime_features_tool,
    create_text_features_tool,
    encode_categorical_features_tool,
    
    # Data generation tools
    generate_synthetic_data_tool,
    balance_dataset_tool,
    infer_data_schema_tool,
    
    # Time series tools
    detect_time_series_patterns_tool,
    create_time_series_features_tool,
    fill_missing_time_series_tool
)

# Also import functions directly for easier use
from dataops_agent.tools.data_processing.profiling_tools import profile_dataframe_comprehensive
from dataops_agent.tools.data_processing.generation_tools import generate_synthetic_data
from dataops_agent.tools.data_processing.feature_engineering_tools import create_polynomial_features
from dataops_agent.tools.data_processing.timeseries_tools import detect_time_series_patterns

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
    
    # Create datetime features
    print("\n2. Creating datetime features...")
    datetime_df = create_datetime_features(df, ['join_date'])
    datetime_cols = [col for col in datetime_df.columns if 'join_date_' in col]
    print(f"  Created {len(datetime_cols)} datetime features:")
    print(f"  Sample features: {datetime_cols[:5]}")
    
    # Create text features (if we had text columns)
    print("\n3. Text feature extraction...")
    # Convert name to text features for demo
    text_df = create_text_features(df, ['name'])
    text_cols = [col for col in text_df.columns if 'name_' in col]
    print(f"  Created {len(text_cols)} text features:")
    print(f"  Text features: {text_cols}")
    
    return poly_df


def demo_synthetic_data_generation():
    """Demonstrate synthetic data generation."""
    print("\n=== SYNTHETIC DATA GENERATION DEMO ===")
    
    # Define schema for synthetic data
    schema = {
        'customer_id': {
            'type': 'numeric',
            'params': {'distribution': 'integer', 'low': 1000, 'high': 9999}
        },
        'age': {
            'type': 'numeric', 
            'params': {'distribution': 'normal', 'mean': 35, 'std': 12}
        },
        'income': {
            'type': 'numeric',
            'params': {'distribution': 'normal', 'mean': 50000, 'std': 15000}
        },
        'gender': {
            'type': 'categorical',
            'params': {'categories': ['Male', 'Female', 'Other'], 'probabilities': [0.45, 0.45, 0.1]}
        },
        'city': {
            'type': 'categorical',
            'params': {'categories': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']}
        },
        'signup_date': {
            'type': 'datetime',
            'params': {'start_date': '2020-01-01', 'end_date': '2023-12-31'}
        },
        'name': {
            'type': 'text',
            'params': {'text_type': 'name'}
        },
        'email': {
            'type': 'text',
            'params': {'text_type': 'email'}
        },
        'is_premium': {
            'type': 'boolean',
            'params': {'prob_true': 0.3}
        }
    }
    
    # Generate synthetic data
    print("\n1. Generating synthetic customer data...")
    synthetic_df = generate_synthetic_data(schema, n_rows=500, seed=42)
    print(f"  Generated dataset shape: {synthetic_df.shape}")
    print(f"  Columns: {list(synthetic_df.columns)}")
    print("\n  Sample rows:")
    print(synthetic_df.head(3).to_string())
    
    return synthetic_df


def demo_time_series_analysis():
    """Demonstrate time series analysis capabilities."""
    print("\n=== TIME SERIES ANALYSIS DEMO ===")
    
    # Create sample time series data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_days = len(dates)
    
    # Create realistic time series with trend, seasonality, and noise
    trend = np.linspace(100, 200, n_days)
    seasonal = 50 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)  # Annual seasonality
    weekly = 20 * np.sin(2 * np.pi * np.arange(n_days) / 7)        # Weekly seasonality
    noise = np.random.normal(0, 10, n_days)
    
    sales = trend + seasonal + weekly + noise
    website_visits = trend * 2 + seasonal * 1.5 + np.random.normal(0, 20, n_days)
    
    ts_df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'website_visits': website_visits,
        'temperature': 20 + 15 * np.sin(2 * np.pi * np.arange(n_days) / 365.25) + np.random.normal(0, 3, n_days)
    })
    
    print(f"\n1. Time series dataset shape: {ts_df.shape}")
    print(f"   Date range: {ts_df['date'].min()} to {ts_df['date'].max()}")
    
    # Detect patterns
    print("\n2. Detecting time series patterns...")
    patterns = detect_time_series_patterns(
        ts_df, 
        datetime_column='date', 
        value_columns=['sales', 'website_visits', 'temperature']
    )
    
    for column, pattern_info in patterns.items():
        print(f"\n   {column.upper()}:")
        print(f"     Trend direction: {pattern_info['trend']['direction']}")
        print(f"     Trend strength: {pattern_info['trend']['strength']:.4f}")
        print(f"     Has seasonality: {pattern_info['seasonality']['has_seasonality']}")
        if pattern_info['seasonality']['detected_periods']:
            print(f"     Detected periods: {pattern_info['seasonality']['detected_periods'][:3]}")
        print(f"     Anomalies: {pattern_info['anomalies']['count']} ({pattern_info['anomalies']['percentage']:.2f}%)")
        print(f"     Is stationary: {pattern_info['stationarity']['is_stationary']}")
    
    return ts_df


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
    
    # Demo 3: Synthetic Data Generation
    synthetic_df = demo_synthetic_data_generation()
    
    # Demo 4: Time Series Analysis
    ts_df = demo_time_series_analysis()
    
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
