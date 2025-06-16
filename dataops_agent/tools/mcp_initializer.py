# web_search_agent/tools/mcp_initializer.py

import os
import threading
import asyncio
from dotenv import load_dotenv
import atexit
from typing import Optional

load_dotenv()

# Global variables for managing MCP tool initialization state
_mcp_tools = None
_exit_stack = None
_initialized = False
_initialization_in_progress = False
_init_lock = threading.Lock()

def _set_agent_mcp_tools(tools, initialized_status):
    """
    Internal function to set MCP tools for agents to avoid circular imports.
    """
    try:
        # Import the set_mcp_tools functions here to avoid circular imports
        from dataops_agent.agents.source_discovery_agent import set_mcp_tools as set_source_discovery_mcp_tools
        from dataops_agent.agents.data_extraction_agent import set_mcp_tools as set_data_extraction_mcp_tools
        
        set_source_discovery_mcp_tools(tools, initialized_status)
        set_data_extraction_mcp_tools(tools, initialized_status)
    except ImportError as e:
        print(f"Warning: Could not set MCP tools for some agents: {e}")

async def initialize_mcp_tools(root_agent_instance=None):
    """
    Initializes MCP tools for the agents, specifically for the researcher agent.
    This function ensures that MCP tools are initialized only once and handles
    the connection and cleanup of the MCP server.

    Args:
        root_agent_instance: The root SequentialAgent instance to which sub-agents
                             are attached. Used to assign tools to the researcher agent.
    """
    global _mcp_tools, _exit_stack, _initialized, _initialization_in_progress
    
    if _initialized:
        return _mcp_tools
    
    with _init_lock:
        if _initialized:
            return _mcp_tools
            
        if _initialization_in_progress:
            # If initialization is already in progress, wait for it to complete
            while _initialization_in_progress:
                await asyncio.sleep(0.1)
            return _mcp_tools
        
        _initialization_in_progress = True
    
    try:
        from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
        
        print("Connecting to Bright Data MCP...")
        # Establish connection to the Bright Data MCP server
        mcp = MCPToolset(
            connection_params=StdioServerParameters(
                command='npx',
                args=["-y", "@brightdata/mcp"],
                env={
                    "API_TOKEN": os.getenv("BRIGHTDATA_API_TOKEN", ""),
                    "WEB_UNLOCKER_ZONE": os.getenv("BRIGHTDATA_UB_ZONE", ""),
                    "BROWSER_AUTH": os.getenv("BRIGHTDATA_BROWSER_AUTH", ""),
                }
            )
        )

        tools, exit_stack = await mcp.get_tools(), mcp.close

        print(f"MCP Toolset created successfully with {len(tools)} tools")

        _mcp_tools = tools
        _exit_stack = exit_stack

        # Register a cleanup function to close the MCP server connection on exit
        def cleanup_mcp():
            global _exit_stack
            if _exit_stack:
                print("Closing MCP server connection...")
                try:
                    # Create a new event loop for cleanup if the main one is closed
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(_exit_stack())
                    loop.close()
                    print("MCP server connection closed successfully.")
                except Exception as e:
                    print(f"Error closing MCP connection: {e}")
                finally:
                    _exit_stack = None
        
        atexit.register(cleanup_mcp)
        
        _initialized = True
        
        # Assign the initialized tools to the relevant agents
        if root_agent_instance:
            for agent in root_agent_instance.sub_agents:
                if agent.name == "source_discovery_agent":
                    agent.tools = tools
                    print(f"Successfully added {len(tools)} tools to source_discovery_agent")
                elif agent.name == "data_extraction_agent":
                    agent.tools = tools
                    print(f"Successfully added {len(tools)} tools to data_extraction_agent")
            
            # List some tool names for debugging
            tool_names = [tool.name for tool in tools[:5]]
            print(f"Available tools include: {', '.join(tool_names)}")
        
        # Set MCP tools for agents (avoiding circular import)
        _set_agent_mcp_tools(tools, True)
                
        print("MCP initialization complete!")
        return tools
        
    except Exception as e:
        print(f"Error initializing MCP tools: {e}")
        return None
    finally:
        _initialization_in_progress = False

async def wait_for_initialization(root_agent_instance=None):
    """
    Waits for MCP initialization to complete. If not already initialized,
    it triggers the initialization process.

    Args:
        root_agent_instance: The root SequentialAgent instance.
    """
    global _initialized
    
    if not _initialized:
        print("Starting initialization in callback...")
        await initialize_mcp_tools(root_agent_instance)
    
    return _initialized

def get_mcp_initialization_status():
    """Returns the current initialization status of MCP tools."""
    global _initialized
    return _initialized

def get_mmcp_tools():
    """Returns the initialized MCP tools."""
    global _mcp_tools
    return _mcp_tools
