from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient
from typing import Dict, Any
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage
from IPython.display import Markdown
from pprint import pprint
import os

os.environ["LANGCHAIN_PROJECT"] = "lca-lc-foundation"

load_dotenv()

@tool
def tavily_search(query: str) -> Dict[str, Any]:
    """Search the web for information."""
    tavily = TavilyClient()
    return tavily.search(query)

system_prompt = """

    You are a personal chef. The user will give you a list of ingredients they have left over in their house.
    Using the web search tool, search the web for recipes that can be made with the ingredients they have.
    Return recipe suggestions and eventually the recipe instructions to the user, if requested.

"""

agent = create_agent(
    model="gpt-5-nano",
    tools=[tavily_search],
    system_prompt=system_prompt,
    checkpointer=InMemorySaver(),
)

#  follow this: https://docs.langchain.com/langsmith/local-server