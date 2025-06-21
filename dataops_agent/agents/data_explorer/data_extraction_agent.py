from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types
from typing import List, Dict, Any, Optional
import asyncio
import time, os
import asyncio
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

DATA_EXTRACTION_AI_MODEL =  "gemini-2.5-flash-preview-05-20"


data_extraction_agent = LlmAgent(
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
    ]
  )
