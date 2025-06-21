from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types
from typing import List, Dict, Any, Optional
import asyncio
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.adk.tools import google_search
import os

SOURCE_DISCOVERY_AI_MODEL = "gemini-2.0-flash"

# Global variables for MCP tool initialization (will be managed by mcp_initializer)
_mcp_tools = None
_initialized = False

def set_mcp_tools(tools, initialized_status):
    """
    Sets the MCP tools and initialization status for the source discovery agent.
    This function is called by the MCP initializer to provide the necessary tools.
    """
    global _mcp_tools, _initialized
    _mcp_tools = tools
    _initialized = initialized_status

def check_source_discovery_tools(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    """
    Callback function to check if source discovery tools are initialized.
    If not, it triggers the MCP initialization and asks the user to retry.
    """
    # Import initialize_mcp_tools and get_mcp_initialization_status here to avoid circular dependency at module level
    from dataops_agent.tools.mcp_initializer import initialize_mcp_tools, get_mcp_initialization_status

    global _mcp_tools, _initialized
    
    agent_name = callback_context.agent_name
    
    root_agent_instance = getattr(callback_context, 'root_agent_instance', None)

    if agent_name == "source_discovery_agent" and not get_mcp_initialization_status():
        print("Source Discovery agent needs tools - will start initialization")
        
        loop = asyncio.get_event_loop()
        loop.create_task(initialize_mcp_tools(root_agent_instance=root_agent_instance))
        
        print("Initialization started in background. Asking user to retry.")
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="Initializing source discovery tools. This happens only once. Please try your query again in a few moments.")]
            )
        )
    
    return None

def create_data_explorer_agent():
    return Agent(
        name="data_explorer_agent",
        description="Identifies and prioritizes web sources for data extraction.",
        model=SOURCE_DISCOVERY_AI_MODEL,
        instruction="""
        Given a user query, your task is to identify and prioritize the most relevant web sources for data extraction. Consider both general web research (e.g., using search engines) and specialized domains or websites that may contain authoritative or high-quality data.

        You may use tools such as `search_engine` to perform web searches and list relevant sites, and `scraping_browser_get_text` to extract content from those sites. Analyze the search results to determine which sources are most likely to contain the required data. If the query suggests that specialized or domain-specific sources are needed, identify and include those as well.

        For each discovered source, provide:
        - The source URL or name
        - Whether it is a general web research or specific site source
        - Also a brief description of what type of data is expected from this source

        Be thorough and creative in your approach, considering multiple strategies to ensure comprehensive source discovery.

        use:
        - `scraping_browser_get_text` when you need to retrieve the visible text content of the current page.
        - `search_engine` to perform web searches and list relevant sites.
        """,
        tools=[
            MCPToolset(
                connection_params=StdioServerParameters(
                    command='npx',
                    args=["-y", "@brightdata/mcp"],
                    env={
                        "API_TOKEN": os.getenv("BRIGHTDATA_API_TOKEN", ""),
                        "WEB_UNLOCKER_ZONE": os.getenv("BRIGHTDATA_UB_ZONE", ""),
                        "BROWSER_AUTH": os.getenv("BRIGHTDATA_BROWSER_AUTH", ""),
                    }
                )
            ),
            google_search
        ],
        # before_model_callback=check_source_discovery_tools
    )
