from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types
from typing import List, Dict, Any, Optional
import asyncio
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.adk.tools import google_search, FunctionTool
import os
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.code_executors import BuiltInCodeExecutor


DATA_PROCESSING_AI_MODEL = "gemini-2.0-flash"



code_agent = LlmAgent(
    name='code_agent',
    model=DATA_PROCESSING_AI_MODEL,
    executor=[BuiltInCodeExecutor],
    instruction="""You are a Python code execution agent. Your task is to execute Python code to process data.
        You will receive Python code as input, and you should execute it to process the data as specified.
        Ensure that the code is safe to execute and does not contain any harmful operations.
        If the code requires any specific libraries, ensure they are available in the execution environment.
    """,
    description="Executes Python code to process data",
)

data_processor_agent = LlmAgent(
        name="data_processor_agent",
        description="",
        model=DATA_PROCESSING_AI_MODEL,
        instruction="""
        
        """,
        sub_agents=[
            code_agent
        ],
        tools=[
            
        ],
    )


