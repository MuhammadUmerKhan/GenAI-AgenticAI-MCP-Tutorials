# Wedding Planner Multi-Agent System (LangGraph)

A **fully white-box, hierarchical, production-grade multi-agent system** built with **LangGraph** for planning destination weddings.

This project demonstrates state-of-the-art agent architecture:
- Each specialist (Travel, Venue, Playlist) is a **standalone, reusable sub-graph**
- The main **Coordinator** is a ReAct agent that dynamically calls sub-graphs via tools
- Full control, observability, async support, and extensibility
- No black-box `AgentExecutor` or legacy patterns

## Architecture Overview

```
Main Graph (Coordinator)
├── Uses LLM with 3 tools → each tool calls a sub-graph
├── ReAct loop via ToolNode + tools_condition
└── Aggregates results into final wedding plan

Sub-Graphs (Independent & Reusable)
├── Travel Agent Sub-Graph → Flight search via Kiwi.com MCP API
├── Venue Agent Sub-Graph → Web search via Tavily
└── Playlist Agent Sub-Graph → SQL queries on Chinook.db
```

Each sub-graph has:
- Its own state
- Tool calling with `bind_tools`
- Full ReAct loop (`ToolNode` + `tools_condition`)
- Clean message handling

## Features

- **Hierarchical agents** (sub-graphs as tools)
- **Full async support**
- **White-box control** — every step visible and modifiable
- **Reusable sub-agents** — can be used in other projects
- **LangSmith-ready** — perfect tracing of all agent steps
- **Extensible** — easy to add catering, photography, budget agents

## Requirements

```bash
pip install langgraph langchain langchain-openai langchain-mcp-adapters tavily-python python-dotenv sqlalchemy
```

## Environment Variables (.env)

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
# Add other keys if needed
```

## Project Structure

```
.
├── main.py                  # Complete code (provided below)
├── prompts.py               # System prompts for all agents
├── resources/Chinook.db     # Sample music database
└── .env
```

## prompts.py Example

```python
coordinator_system_prompt = """
You are a professional wedding coordinator. Your job is to:
1. Extract origin, destination, guest count, and music genre from the user.
2. Delegate specialized tasks to the correct agents using tools.
3. Once all information is gathered, synthesize a complete wedding plan.
"""

travel_agent_system_prompt = """
You are an expert travel agent for destination weddings...
"""

venue_agent_system_prompt = """
You are a wedding venue specialist...
"""

playlist_agent_system_prompt = """
You are a music curator for weddings...
"""
```

## How to Run

```python
import asyncio
from main import main_coordinator_agent

async def main():
    app = await main_coordinator_agent()
    
    result = await app.ainvoke({
        "messages": [HumanMessage(content="I'm from London and I'd like a wedding in Paris for 100 guests with jazz music")],
        # Optional: pre-fill if testing
        # "origin": None, "destination": None, etc.
    })
    
    print(result["messages"][-1].content)

asyncio.run(main())
```

## Why This Architecture is Advanced

- **No legacy `create_agent`** → full control
- **Sub-graphs as tools** → dynamic routing by LLM
- **Proper ReAct loops** in every agent
- **Async everywhere** → production ready
- **Observable** → perfect for LangSmith debugging

This pattern is used in real-world enterprise agent systems.

## Future Improvements

- Add structured output (Pydantic) for state extraction
- Add human-in-the-loop approval nodes
- Add persistence with `SqliteSaver`
- Add streaming with `astream_events()`
- Add error recovery and retries

## Credits

Built with ❤️ using LangGraph.
Inspired by advanced multi-agent patterns from [LangChain team](https://academy.langchain.com/courses/take/foundation-introduction-to-langchain-python/lessons/71234864-lesson-4-wedding-planner-project).
```