import os, asyncio
from dotenv import load_dotenv
from google.adk.agents import SequentialAgent, Agent
from datetime import datetime
from typing import Dict, List
from google.adk.agents.callback_context import CallbackContext

# Import agent creation functions from their new modules
from dataops_agent.agents import (
    create_query_analyzer_agent,
    create_source_discovery_agent,
    create_data_extraction_agent,
    create_schema_generator_agent,
)

# Import MCP initialization function
from dataops_agent.tools.mcp_initializer import initialize_mcp_tools, get_mcp_initialization_status

load_dotenv()

print("Module loaded: dataset_generator_agent")

# Create the root SequentialAgent instance
root_agent = SequentialAgent(
    name="dataset_generator_agent",
    description="A real-time data set generator AI agent that can be used to generate real-time datasets by scraping website content for AI ML projects on demand by the users.",
    sub_agents=[
        create_query_analyzer_agent(),
        create_source_discovery_agent(),
        create_data_extraction_agent(),
        create_schema_generator_agent(),
    ],
)

# Initialize the MCP tools for the root agent
async def setup_tools_for_agent():
    await initialize_mcp_tools(root_agent_instance=root_agent)
async def _initialize_agent_tools():
    if not get_mcp_initialization_status():
        await setup_tools_for_agent()

try:
    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop.create_task(_initialize_agent_tools())
    else:
        asyncio.run(_initialize_agent_tools())
except RuntimeError:
    # This handles cases where there's no running loop, but also no loop set for the current thread
    asyncio.run(_initialize_agent_tools())

print("Agent structure created. MCP tools will be initialized.")
