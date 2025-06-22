# Data Processing Functions Documentation

This document lists all the data processing functions available in the project, now refactored into multiple specialized modules within `dataops_agent/tools/data_processing/`.

## Table of Contents
- [Data Processing Functions Documentation](#data-processing-functions-documentation)
  - [Table of Contents](#table-of-contents)
  - [Core Data Processing Classes](#core-data-processing-classes)
    - [DataCleaner](#datacleaner)
    - [TextProcessor](#textprocessor)
    - [NumericProcessor](#numericprocessor)
    - [DateTimeProcessor](#datetimeprocessor)
    - [ContactProcessor](#contactprocessor)
    - [MissingDataHandler](#missingdatahandler)
  - [Pipeline Functions (`cleaning_tools.py`)](#pipeline-functions-cleaning_toolspy)
  - [Data Validation Functions (`validation_tools.py`)](#data-validation-functions-validation_toolspy)
  - [Data Profiling Functions (`profiling_tools.py`)](#data-profiling-functions-profiling_toolspy)
  - [Feature Engineering Functions (`feature_engineering_tools.py`)](#feature-engineering-functions-feature_engineering_toolspy)
  - [Google ADK Tools](#google-adk-tools)

---

## Core Data Processing Classes

### DataCleaner
- `log_operation(operation: str, details: str = "")`  
  Log cleaning operations for audit trail.
- `get_cleaning_report() -> pd.DataFrame`  
  Get a report of all cleaning operations performed.

### TextProcessor
- `fix_encoding(text: str) -> str`  
  Fix broken Unicode encoding using ftfy.
- `normalize_unicode(text: str, remove_accents: bool = False) -> str`  
  Normalize Unicode text and optionally remove accents.
- `clean_whitespace(text: str) -> str`  
  Clean various types of whitespace characters.
- `remove_html_tags(text: str) -> str`  
  Remove HTML tags from text.
- `standardize_case(text: str, case_type: str = 'lower') -> str`  
  Standardize text case (lower, upper, title, sentence).

### NumericProcessor
- `extract_numbers(text: str) -> Optional[float]`  
  Extract numeric values from text.
- `handle_outliers(data: pd.Series, method: str = 'iqr', factor: float = 1.5) -> pd.Series`  
  Handle outliers using IQR or z-score methods.
- `normalize_data(data: pd.Series, method: str = 'standard') -> pd.Series`  
  Normalize numeric data (standard, minmax, robust).

### DateTimeProcessor
- `parse_flexible_date(date_str: str) -> Optional[pd.Timestamp]`  
  Parse dates in various formats using dateutil.
- `standardize_date_format(date_series: pd.Series, target_format: str = '%Y-%m-%d') -> pd.Series`  
  Standardize date formats in a series.
- `extract_date_components(date_series: pd.Series) -> pd.DataFrame`  
  Extract date components (year, month, day, etc.).

### ContactProcessor
- `parse_phone_number(phone: str, country_code: str = 'US') -> Dict[str, Any]`  
  Parse and validate phone numbers.
- `clean_email(email: str) -> str`  
  Clean and validate email addresses.

### MissingDataHandler
- `analyze_missing_data(df: pd.DataFrame) -> Dict[str, Any]`  
  Analyze missing data patterns.
- `visualize_missing_data(df: pd.DataFrame, save_path: Optional[str] = None)`  
  Visualize missing data patterns using missingno.
- `impute_missing_values(df: pd.DataFrame, strategy: str = 'mean', columns: Optional[List[str]] = None) -> pd.DataFrame`  
  Impute missing values using various strategies.

## Pipeline Functions (`cleaning_tools.py`)
- `clean_dataframe_comprehensive(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]`  
  Comprehensive data cleaning pipeline.
- `quick_clean(df: pd.DataFrame) -> pd.DataFrame`  
  Quick data cleaning with sensible defaults.

## Data Validation Functions (`validation_tools.py`)
- `validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]`  
  Validate data quality and provide recommendations.

## Data Profiling Functions (`profiling_tools.py`)
- `profile_dataframe_comprehensive(df: pd.DataFrame) -> Dict[str, Any]`  
  Generate comprehensive profile including statistical summaries, data types, missing values, duplicates, and distribution analysis.
- `detect_data_types_advanced(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]`  
  Advanced data type detection that goes beyond pandas' basic inference (emails, phone numbers, URLs, etc.).
- `analyze_data_distributions(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]`  
  Analyze statistical distributions of numeric columns using various statistical tests.
- `generate_data_quality_score(df: pd.DataFrame) -> Dict[str, Any]`  
  Generate overall data quality score based on completeness, uniqueness, consistency, and validity.
- `compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, name1: str = "Dataset 1", name2: str = "Dataset 2") -> Dict[str, Any]`  
  Compare two DataFrames and identify differences in structure and content.

## Feature Engineering Functions (`feature_engineering_tools.py`)
- `create_polynomial_features(df: pd.DataFrame, columns: List[str], degree: int = 2, interaction_only: bool = False) -> pd.DataFrame`  
  Create polynomial features from specified numeric columns.
- `create_binning_features(df: pd.DataFrame, column: str, bins: Union[int, List[float]], labels: Optional[List[str]] = None) -> pd.Series`  
  Create binned categorical features from continuous variables.
- `create_aggregation_features(df: pd.DataFrame, group_columns: List[str], agg_columns: List[str], agg_functions: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame`  
  Create aggregation features based on grouping columns.
- `create_datetime_features(df: pd.DataFrame, datetime_columns: List[str]) -> pd.DataFrame`  
  Extract comprehensive datetime features including cyclical encoding.
- `create_text_features(df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame`  
  Extract features from text columns (length, word count, character types, etc.).
- `encode_categorical_features(df: pd.DataFrame, encoding_config: Dict[str, Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Any]]`  
  Apply various encoding techniques (one-hot, label, ordinal, frequency).
- `select_features_statistical(df: pd.DataFrame, target_column: str, task_type: str = 'classification', k: int = 10) -> Tuple[List[str], Dict[str, float]]`  
  Select features using statistical tests (F-test, chi-square).
- `select_features_model_based(df: pd.DataFrame, target_column: str, task_type: str = 'classification', max_features: Optional[int] = None) -> Tuple[List[str], Dict[str, float]]`  
  Select features using machine learning models (Random Forest importance).
- `apply_dimensionality_reduction(df: pd.DataFrame, method: str = 'pca', n_components: Optional[int] = None, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]`  
  Apply dimensionality reduction techniques (PCA, t-SNE).


## Google ADK Tools

All functions above are available as Google ADK Function Tools with the suffix `_tool`. For example:
- `clean_dataframe_comprehensive_tool`
- `profile_dataframe_comprehensive_tool`
- `generate_synthetic_data_tool`
- `detect_time_series_patterns_tool`

The complete collection is available as `ALL_DATA_PROCESSING_TOOLS` list in `data_processing_tools.py`.

---

**Dependencies:**
- **Required**: pandas, numpy, scipy, scikit-learn
- **Optional**: ftfy, pyjanitor, phonenumbers, missingno, seaborn, matplotlib, faker, dask
- Some functions require optional dependencies for full functionality. Fallback implementations are provided where possible.

**Note:**
See the source code for detailed docstrings and usage examples. Each function includes comprehensive error handling and logging.
