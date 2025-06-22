
# Available functionality:
# - clean_dataframe_comprehensive(): Complete cleaning pipeline
# - validate_data_quality(): Data quality assessment
# - profile_dataframe_comprehensive(): Comprehensive data profiling
# - create_polynomial_features(): Feature engineering with polynomials
# - generate_synthetic_data(): Generate synthetic datasets
# - detect_time_series_patterns(): Time series pattern analysis

from google.adk.tools import FunctionTool

# Import core cleaning and validation tools
from .cleaning_tools import clean_dataframe_comprehensive, quick_clean
from .validation_tools import validate_data_quality

# Import new advanced tools
from .profiling_tools import (
    profile_dataframe_comprehensive,
    detect_data_types_advanced,
    analyze_data_distributions,
    generate_data_quality_score,
    compare_dataframes
)

from .feature_engineering_tools import (
    create_polynomial_features,
    create_binning_features,
    create_aggregation_features,
    create_datetime_features,
    create_text_features,
    encode_categorical_features,
    select_features_statistical,
    select_features_model_based,
    apply_dimensionality_reduction
)

from .generation_tools import (
    generate_synthetic_data,
    augment_data_with_noise,
    generate_smote_samples,
    create_time_series_variations,
    expand_categorical_combinations,
    balance_dataset,
    infer_data_schema
)

from .timeseries_tools import (
    detect_time_series_patterns,
    create_time_series_features,
    detect_change_points,
    decompose_time_series,
    create_time_series_clusters,
    fill_missing_time_series,
    calculate_time_series_similarity,
    generate_time_series_forecast_features
)

# Create Google ADK Function Tools
# Core tools
clean_dataframe_comprehensive_tool = FunctionTool(func=clean_dataframe_comprehensive)
quick_clean_tool = FunctionTool(func=quick_clean)
validate_data_quality_tool = FunctionTool(func=validate_data_quality)

# Profiling tools
profile_dataframe_comprehensive_tool = FunctionTool(func=profile_dataframe_comprehensive)
detect_data_types_advanced_tool = FunctionTool(func=detect_data_types_advanced)
analyze_data_distributions_tool = FunctionTool(func=analyze_data_distributions)
generate_data_quality_score_tool = FunctionTool(func=generate_data_quality_score)
compare_dataframes_tool = FunctionTool(func=compare_dataframes)

# Feature engineering tools
create_polynomial_features_tool = FunctionTool(func=create_polynomial_features)
create_binning_features_tool = FunctionTool(func=create_binning_features)
create_aggregation_features_tool = FunctionTool(func=create_aggregation_features)
create_datetime_features_tool = FunctionTool(func=create_datetime_features)
create_text_features_tool = FunctionTool(func=create_text_features)
encode_categorical_features_tool = FunctionTool(func=encode_categorical_features)
select_features_statistical_tool = FunctionTool(func=select_features_statistical)
select_features_model_based_tool = FunctionTool(func=select_features_model_based)
apply_dimensionality_reduction_tool = FunctionTool(func=apply_dimensionality_reduction)

# Data generation tools
generate_synthetic_data_tool = FunctionTool(func=generate_synthetic_data)
augment_data_with_noise_tool = FunctionTool(func=augment_data_with_noise)
generate_smote_samples_tool = FunctionTool(func=generate_smote_samples)
create_time_series_variations_tool = FunctionTool(func=create_time_series_variations)
expand_categorical_combinations_tool = FunctionTool(func=expand_categorical_combinations)
balance_dataset_tool = FunctionTool(func=balance_dataset)
infer_data_schema_tool = FunctionTool(func=infer_data_schema)

# Time series tools
detect_time_series_patterns_tool = FunctionTool(func=detect_time_series_patterns)
create_time_series_features_tool = FunctionTool(func=create_time_series_features)
detect_change_points_tool = FunctionTool(func=detect_change_points)
decompose_time_series_tool = FunctionTool(func=decompose_time_series)
create_time_series_clusters_tool = FunctionTool(func=create_time_series_clusters)
fill_missing_time_series_tool = FunctionTool(func=fill_missing_time_series)
calculate_time_series_similarity_tool = FunctionTool(func=calculate_time_series_similarity)
generate_time_series_forecast_features_tool = FunctionTool(func=generate_time_series_forecast_features)

# Collection of all tools for easy import
ALL_DATA_PROCESSING_TOOLS = [
    # Core tools
    clean_dataframe_comprehensive_tool,
    quick_clean_tool,
    validate_data_quality_tool,
    
    # Profiling tools
    profile_dataframe_comprehensive_tool,
    detect_data_types_advanced_tool,
    analyze_data_distributions_tool,
    generate_data_quality_score_tool,
    compare_dataframes_tool,
    
    # Feature engineering tools
    create_polynomial_features_tool,
    create_binning_features_tool,
    create_aggregation_features_tool,
    create_datetime_features_tool,
    create_text_features_tool,
    encode_categorical_features_tool,
    select_features_statistical_tool,
    select_features_model_based_tool,
    apply_dimensionality_reduction_tool,
    
    # Data generation tools
    generate_synthetic_data_tool,
    augment_data_with_noise_tool,
    generate_smote_samples_tool,
    create_time_series_variations_tool,
    expand_categorical_combinations_tool,
    balance_dataset_tool,
    infer_data_schema_tool,
    
    # Time series tools
    detect_time_series_patterns_tool,
    create_time_series_features_tool,
    detect_change_points_tool,
    decompose_time_series_tool,
    create_time_series_clusters_tool,
    fill_missing_time_series_tool,
    calculate_time_series_similarity_tool,
    generate_time_series_forecast_features_tool
]
