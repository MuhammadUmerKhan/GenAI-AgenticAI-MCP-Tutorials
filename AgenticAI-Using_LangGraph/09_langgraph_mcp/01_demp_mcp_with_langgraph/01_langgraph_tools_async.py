from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import tool
import sqlite3, requests, os, asyncio
from dotenv import load_dotenv

load_dotenv()

# -------------------
# 1. LLM
# -------------------
# llm = ChatGroq(model="openai/gpt-oss-20b")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# -------------------
# 2. Tools
# -------------------
# Tools
search_tool = DuckDuckGoSearchRun(region="us-en")

SERVERS = {
    "Arithmetic Math Calculator": {
        "command": "/home/muhammadumerkhan/.local/bin/uv",
        "args": [
            "run",
            "--with",
            "fastmcp",
            "fastmcp",
            "run",
            "/home/muhammadumerkhan/Gen-Agentic-AI-Tutorials/AgenticAI-Using_LangGraph/09_langgraph_mcp/01_demp_mcp_with_langgraph/01_mcp_servers/01_math_server.py"
        ],
        "env": {},
        "transport": "stdio"
    },
    "Expense Tracker": {
      "command": "/home/muhammadumerkhan/.local/bin/uv",
      "args": [
        "run",
        "--with",
        "fastmcp",
        "fastmcp",
        "run",
        "/home/muhammadumerkhan/Gen-Agentic-AI-Tutorials/MCP_tutorial/01_Local_MCP_Server/02_expense_tracker_server/01_expense_tracker.py"
      ],
      "env": {},
      "transport": "stdio"
    }
}

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 4. Graph
# -------------------
async def build_graph():
    
    client = MultiServerMCPClient(SERVERS)
    
    tools = await client.get_tools()

    named_tools = {tool.name: tool for tool in tools}
    print("Available tools:", list(named_tools.keys()))

    llm_with_tools = llm.bind_tools(tools)
    
    async def chat_node(state: ChatState):
        """LLM node that may answer or request a tool call."""
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")

    graph.add_conditional_edges("chat_node",tools_condition)
    graph.add_edge('tools', 'chat_node')

    chatbot = graph.compile()

    return chatbot
    

async def main():
    chatbot = await build_graph()
    
    result = await chatbot.ainvoke(
        {"messages": [HumanMessage(content="show my expenses this year 2025")]},
    )
    
    print(result['messages'][-1].content)

if __name__ == "__main__":
    asyncio.run(main())