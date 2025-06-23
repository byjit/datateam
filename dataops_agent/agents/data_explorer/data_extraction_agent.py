from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types
from typing import List, Dict, Any, Optional
import asyncio
import time, os
import asyncio

from grpc import server
from dataops_agent.tools.web_tools import download_kaggle_dataset_tool, save_to_local_file_tool
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioConnectionParams, StdioServerParameters

DATA_EXTRACTION_AI_MODEL =  "gemini-2.5-flash"


data_extraction_agent = LlmAgent(
    name="data_extraction_agent",
    description="Can scrape data from any sources, of any type. Can download kaggle datasets too.",
    model=DATA_EXTRACTION_AI_MODEL,
    output_key="extracted_data",
    instruction="""
    You are a data extraction agent responsible for gathering data required for dataset generation. 

    You will be provided with some information. You need to analyze this information and make a plan to extract the data from the given source.
    If it's a kaggle dataset, you will download it using the provided tool `download_kaggle_dataset_tool`. 
    
    If it's any other web source, you will use the appropriate web scraping tools to extract the data.
    
    Steps:
    1. Analyze the provided information to determine the type of source and the expected data.
    2. Select the appropriate tools based on the type of source:
    - If the source is a kaggle dataset, use `download_kaggle_dataset_tool`.
    - If the source hints at "amazon", use:
      - `web_data_amazon_product`
      - `web_data_amazon_product_reviews`
    - If the source hints at "linkedin", use:
      - `web_data_linkedin_person_profile`
      - `web_data_linkedin_company_profile`
    - If the data type is related to "social_media", use:
      - `web_data_instagram_profiles`
      - `web_data_x_posts`
      - `web_data_facebook_posts`
    - If the source is another specific site, plan on how to scrape it using the following tools (can be used in combination with multi-turns):
      1. `scraping_browser_navigate` (to visit the URL)
      2. `scraping_browser_get_text` (to extract content)
      3. `scraping_browser_links` and `scraping_browser_click` (to navigate within the site if needed)
    - If the source is general web research, use the `search_engine` tool.
    3. Extract the relevant data as described and return it in a structured format suitable for dataset generation.
    4. If the data is extracted/scraped, once it is extracted save it to the local using the `save_to_local_file_tool`.
    5. Tell the user about the status

    Always justify your tool selection and extraction steps. If you encounter issues, provide clear error messages or suggestions for alternative approaches.
    """,
    # before_model_callback=check_data_extraction_tools
    tools=[
        MCPToolset(
            connection_params=StdioConnectionParams(
                server_params= StdioServerParameters(
                    command='npx',
                    args=["-y", "@brightdata/mcp"],
                    env={
                        "API_TOKEN": os.getenv("BRIGHTDATA_API_TOKEN", ""),
                        "WEB_UNLOCKER_ZONE": os.getenv("BRIGHTDATA_UB_ZONE", ""),
                        "BROWSER_AUTH": os.getenv("BRIGHTDATA_BROWSER_AUTH", ""),
                    },
                ),
                timeout=60,  # Set a timeout for the connection
            ),
        ),
        download_kaggle_dataset_tool,
    ]
  )
