from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types
from typing import List, Dict, Any, Optional
import asyncio
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools import google_search, FunctionTool
import os
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.code_executors import BuiltInCodeExecutor

# Import enhanced data processing tools
from dataops_agent.tools.data_processing.data_processing_tools import (
    # Core tools
    clean_dataframe_comprehensive_tool,
    validate_data_quality_tool,
    
    # Profiling tools
    profile_dataframe_comprehensive_tool,
    generate_data_quality_score_tool,
    
    # Feature engineering tools
    create_polynomial_features_tool,
    create_datetime_features_tool,
    encode_categorical_features_tool,
)

DATA_PROCESSING_AI_MODEL = "gemini-2.0-flash"

data_processor_agent = LlmAgent(
    name="data_processor_agent",
    description="Advanced data processing agent with comprehensive cleaning, profiling, feature engineering, and analysis capabilities",
    model=DATA_PROCESSING_AI_MODEL,
    instruction="""
    You are an advanced data processing AI agent specializing in comprehensive data analysis and transformation.
    
    Your capabilities include:
    
    1. **Data Profiling & Quality Assessment**:
       - Use `profile_dataframe_comprehensive_tool` for comprehensive data analysis
       - Use `generate_data_quality_score_tool` to assess data quality
       - Use `validate_data_quality_tool` for detailed validation reports
    
    2. **Data Cleaning & Preprocessing**:
       - Use `clean_dataframe_comprehensive_tool` for thorough data cleaning
       - Handle missing values, duplicates, outliers, and encoding issues
    
    3. **Feature Engineering**:
       - Use `create_polynomial_features_tool` for polynomial feature creation
       - Use `create_datetime_features_tool` for temporal feature extraction
       - Use `encode_categorical_features_tool` for categorical encoding
    
    4. **Code Execution**:
       - Delegate to `code_agent` for custom data processing tasks
       - Execute Python code for complex transformations
    
    **Workflow Approach**:
    1. Always start with data profiling to understand the dataset
    2. Assess data quality and identify issues
    3. Apply appropriate cleaning and preprocessing
    4. Perform feature engineering based on data type and use case
    5. Generate reports and recommendations
    
    **Best Practices**:
    - Provide detailed explanations of your analysis
    - Suggest improvements and next steps
    - Handle errors gracefully with fallback options
    - Document all transformations applied
    """,
    tools=[
        # Core processing tools
        clean_dataframe_comprehensive_tool,
        validate_data_quality_tool,
        
        # Profiling tools
        profile_dataframe_comprehensive_tool,
        generate_data_quality_score_tool,
        
        # Feature engineering tools
        create_polynomial_features_tool,
        create_datetime_features_tool,
        encode_categorical_features_tool,
    ],
)


