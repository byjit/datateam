from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types
from typing import List, Dict, Any, Optional
import asyncio
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.adk.tools import google_search, FunctionTool
from dataops_agent.tools.tools import search_kaggle_datasets_tool, search_sources_using_sonar_tool
import os

SOURCE_DISCOVERY_AI_MODEL = "gemini-2.0-flash"


data_explorer_agent = LlmAgent(
        name="data_explorer_agent",
        description="Helps to discover sources where datasets can be found",
        model=SOURCE_DISCOVERY_AI_MODEL,
        instruction="""
        You are an intelligent bot whose sole purpose is to find sources where datasets can be found.
        Call both the tools provided to you to find the best sources for datasets. And then aggregate the results to provide a final output. And ask the user which source
        would they like to proceed with.

        Present the results in a structured format as a numbered list of sources and the url too if present.
        """,
        tools=[
            search_kaggle_datasets_tool,
            search_sources_using_sonar_tool,
        ],
    )
