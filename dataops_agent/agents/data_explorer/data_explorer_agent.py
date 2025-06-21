from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types
from typing import List, Dict, Any, Optional
import asyncio
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.adk.tools import google_search
from dataops_agent.tools.tools import search_kaggle_datasets, search_using_sonar
import os

SOURCE_DISCOVERY_AI_MODEL = "gemini-2.0-flash"


def create_data_explorer_agent():
    return LlmAgent(
        name="data_explorer_agent",
        description="Helps to discover sources where datasets can be found",
        model=SOURCE_DISCOVERY_AI_MODEL,
        instruction="""
        You are an intelligent bot whose sole purpose is to find sources where datasets can be found.
        Call 
        """,
        tools=[
            
            google_search
        ],
    )
