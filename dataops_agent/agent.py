import os, asyncio
from dotenv import load_dotenv
from google.adk.agents import SequentialAgent, Agent, LlmAgent
from google.adk.tools import agent_tool, FunctionTool
from datetime import datetime
from typing import Dict, List
from google.adk.agents.callback_context import CallbackContext
# Import agent creation functions from their new modules
from dataops_agent.agents.data_explorer import (
    data_explorer_agent,
    data_extraction_agent
)



load_dotenv()


DATAOPS_LLM_MODEL = "gemini-2.5-flash-preview-05-20"

# # Create the root SequentialAgent instance
# data_collector_agent = SequentialAgent(
#     name="dataset_collector_agent",
#     description="A real-time data set generator AI agent that can be used to generate real-time datasets by scraping website content for AI ML projects on demand by the users.",
#     sub_agents=[
#         create_data_extraction_agent(),
#     ]
# )

root_agent = LlmAgent(
    name="coordinator_agent",
    model=DATAOPS_LLM_MODEL,
    instruction="""
    You are a intelligent dataops ai agent whose job is to plan and route the user request to the relevant ai agent according to their capabilities.
    """,
    description="coordinates the workflow of the dataset generation process, ensuring that all sub-agents work together effectively.",
    sub_agents=[
        data_explorer_agent,
    ],
    tools=[]
)