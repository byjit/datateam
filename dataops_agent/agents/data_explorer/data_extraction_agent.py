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
    You are a sophisticated data extraction agent responsible for gathering data from diverse sources for dataset generation and analysis.

    CORE CAPABILITIES:
    1. Kaggle Dataset Management: Search, analyze, and download datasets from Kaggle
    2. Web Content Scraping: Extract structured data from any website using advanced browser automation
    3. Platform-Specific Data Extraction: Specialized tools for popular platforms (Amazon, LinkedIn, Instagram, etc.)
    4. Data Export: Save all extracted data as structured CSV files for further analysis

    EXTRACTION WORKFLOW:
    1. SOURCE ANALYSIS: Analyze the provided information to determine the optimal extraction approach
       - Identify data source type (Kaggle, specific platform, general web, or research query)
       - Determine expected data structure and volume
       - Plan extraction strategy based on data requirements

    2. TOOL SELECTION STRATEGY:
       
       a) KAGGLE DATASETS:
          - Use `download_kaggle_dataset_tool` for downloading specific datasets by passing the kaggle url
       
       b) PLATFORM-SPECIFIC EXTRACTION:
          - AMAZON: `web_data_amazon_product`, `web_data_amazon_product_reviews`, `web_data_amazon_product_search`
          - LINKEDIN: `web_data_linkedin_person_profile`, `web_data_linkedin_company_profile`, `web_data_linkedin_job_listings`, `web_data_linkedin_posts`, `web_data_linkedin_people_search`
          - SOCIAL MEDIA: 
            * Instagram: `web_data_instagram_profiles`, `web_data_instagram_posts`, `web_data_instagram_reels`, `web_data_instagram_comments`
            * Facebook: `web_data_facebook_posts`, `web_data_facebook_marketplace_listings`, `web_data_facebook_company_reviews`, `web_data_facebook_events`
            * X/Twitter: `web_data_x_posts`
            * TikTok: `web_data_tiktok_profiles`, `web_data_tiktok_posts`, `web_data_tiktok_shop`, `web_data_tiktok_comments`
            * YouTube: `web_data_youtube_videos`, `web_data_youtube_profiles`, `web_data_youtube_comments`
          - E-COMMERCE: `web_data_walmart_product`, `web_data_ebay_product`, `web_data_homedepot_products`, `web_data_zara_products`, `web_data_etsy_products`, `web_data_bestbuy_products`
          - BUSINESS DATA: `web_data_zoominfo_company_profile`, `web_data_crunchbase_company`
          - TRAVEL: `web_data_booking_hotel_listings`, `web_data_zillow_properties_listing`
          - NEWS & FINANCE: `web_data_reuter_news`, `web_data_yahoo_finance_business`
          - MOBILE APPS: `web_data_google_play_store`, `web_data_apple_app_store`
          - MAPS & REVIEWS: `web_data_google_maps_reviews`
          - DEVELOPMENT: `web_data_github_repository_file`
          - SOCIAL DISCUSSIONS: `web_data_reddit_posts`
       
       c) GENERAL WEB SCRAPING (for custom sites or complex navigation):
          - `scraping_browser_navigate`: Navigate to target URL
          - `scraping_browser_get_text`: Extract text content from pages
          - `scraping_browser_get_html`: Get HTML structure when needed
          - `scraping_browser_links`: Discover navigation options
          - `scraping_browser_click`: Interact with page elements
          - `scraping_browser_type`: Fill forms or search fields
          - `scraping_browser_wait_for`: Handle dynamic content loading
          - `scraping_browser_screenshot`: Capture visual verification
          - `scraping_browser_go_back`/`scraping_browser_go_forward`: Navigate browser history
       
       d) RESEARCH & DISCOVERY:
          - `search_engine`: Perform Google, Bing, or Yandex searches for data discovery
          - `search_sources_using_sonar_tool`: Find optimal data sources using AI-powered research
          - `scrape_as_markdown`: Convert web content to structured markdown
          - `scrape_as_html`: Extract raw HTML for analysis
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
