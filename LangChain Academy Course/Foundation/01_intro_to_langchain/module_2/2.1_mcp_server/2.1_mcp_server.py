from fastmcp import FastMCP
from tavily import TavilyClient
from typing import Dict, Any
from dotenv import load_dotenv
import requests

load_dotenv()

mcp = FastMCP(name="Web Search")

@mcp.tool(description="Search the web for information.")
def search_web(query: str) -> Dict[str, Any]:
    
    tavily = TavilyClient()
    return tavily.search(query)

@mcp.prompt(description="Analyze data from a langchain-ai repo file with comprehensive insights")
def prompt():
    
    return """
    You are a helpful assistant that answers user questions about LangChain, LangGraph and LangSmith.

    You can use the following tools/resources to answer user questions:
    - search_web: Search the web for information
    - github_file: Access the langchain-ai repo files

    If the user asks a question that is not related to LangChain, LangGraph or LangSmith, you should say "I'm sorry, I can only answer questions about LangChain, LangGraph and LangSmith."

    You may try multiple tool and resource calls to answer the user's question.

    You may also ask clarifying questions to the user to better understand their question.
    """

if __name__ == "__main__":
    mcp.run(transport="stdio")