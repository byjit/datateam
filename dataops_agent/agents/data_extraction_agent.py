from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types
from dataops_agent.tools.mcp_initializer import initialize_mcp_tools, get_mcp_initialization_status
from typing import List, Dict, Any, Optional
import asyncio
import time

DATA_EXTRACTION_AI_MODEL =  "gemini-2.5-flash-preview-05-20"

# Global variables for MCP tool initialization (will be managed by mcp_initializer)
_mcp_tools = None
_initialized = False

def set_mcp_tools(tools, initialized_status):
    """
    Sets the MCP tools and initialization status for the data extraction agent.
    This function is called by the MCP initializer to provide the necessary tools.
    """
    global _mcp_tools, _initialized
    _mcp_tools = tools
    _initialized = initialized_status

def check_data_extraction_tools(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    """
    Callback function to check if data extraction tools are initialized.
    If not, it triggers the MCP initialization and asks the user to retry.
    """
    # Import initialize_mcp_tools and get_mcp_initialization_status here to avoid circular dependency at module level
    from dataops_agent.tools.mcp_initializer import initialize_mcp_tools, get_mcp_initialization_status

    global _mcp_tools, _initialized
    
    agent_name = callback_context.agent_name
    
    root_agent_instance = getattr(callback_context, 'root_agent_instance', None)

    if agent_name == "data_extraction_agent" and not get_mcp_initialization_status():
        print("Data Extraction agent needs tools - will start initialization")
        
        loop = asyncio.get_event_loop()
        loop.create_task(initialize_mcp_tools(root_agent_instance=root_agent_instance))
        
        print("Initialization started in background. Asking user to retry.")
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="Initializing data extraction tools. This happens only once. Please try your query again in a few moments.")]
            )
        )
    
    return None

def create_data_extraction_agent():
    return Agent(
        name="data_extraction_agent",
        description="Executes data extraction using optimal Bright Data MCP tools.",
        model=DATA_EXTRACTION_AI_MODEL,
        output_key="extracted_data",
        instruction="""
        You are a data extraction agent responsible for gathering data required for dataset generation. 

        Given:
        - A discovery source URL
        - The type of discovery source (either "general webresearch" or a specific site)
        - A description of the expected data from this source

        Your tasks:
        1. Analyze the provided information to determine the most appropriate extraction strategy.
        2. Select the correct tool(s) based on the following rules:
            - If the source hints at "amazon", use: 
              - web_data_amazon_product
              - web_data_amazon_product_reviews
            - If the source hints at "linkedin", use:
              - web_data_linkedin_person_profile
              - web_data_linkedin_company_profile
            - If the data type is related to "social_media", use:
              - web_data_instagram_profiles
              - web_data_x_posts
              - web_data_facebook_posts
            - If the source is another specific site:
              - Use browser tools in this order:
                 1. scraping_browser_navigate (to visit the URL)
                 2. scraping_browser_get_text (to extract content)
                 3. scraping_browser_links and scraping_browser_click (to navigate within the site if needed)
            - If the source is general web research:
              - Use the search_engine tool

        3. Extract the relevant data as described and return it in a structured format suitable for dataset generation.

        Always justify your tool selection and extraction steps. If you encounter issues, provide clear error messages or suggestions for alternative approaches.
        """,
        # before_model_callback=check_data_extraction_tools
    )
