from turtle import mode
from google.adk.agents import Agent
from typing import List, Dict, Any

QUERY_ANALYSIS_AI_MODEL = "gemini-2.0-flash"


def create_query_analyzer_agent():
    return Agent(
        name="query_analyzer_agent",
        description="Analyzes user requests and generates structured data extraction plans.",
        model=QUERY_ANALYSIS_AI_MODEL,
        instruction="""
        You are a data extraction planning assistant. Given a user request, your task is to:

        1. Extract and specify the target schema for the requested dataset. For each field, provide:
            - Field name
            - Data type (e.g., string, integer, float, datetime)
            - A brief description of what the field represents

        2. Identify the target query or domain (e.g., ecommerce, professional profiles, social media, etc.) relevant to the user's request.

        Respond in a structured JSON format with two keys:
        - "schema": a list of fields with their name, type, and description
        - "domain": a string describing the target query or domain

        Example output:
        {
          "schema": [
             {"name": "product_name", "type": "string", "description": "Name of the product"},
             {"name": "price", "type": "float", "description": "Current price of the product"},
             {"name": "reviews", "type": "integer", "description": "Number of product reviews"}
          ],
          "domain": "ecommerce"
        }
        """,
        output_key="structured_query"
    )
