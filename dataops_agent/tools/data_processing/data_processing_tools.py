
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
]

__all__ = ['ALL_DATA_PROCESSING_TOOLS']