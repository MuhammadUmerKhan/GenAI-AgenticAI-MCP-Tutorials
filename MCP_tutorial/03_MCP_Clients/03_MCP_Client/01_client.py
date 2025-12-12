import asyncio
import json
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv()

SERVERS = {
    "Arithmetic Math Calculator": {
        "command": "/home/muhammadumerkhan/.local/bin/uv",
        "args": [
            "run",
            "--with",
            "fastmcp",
            "fastmcp",
            "run",
            "/home/muhammadumerkhan/Gen-Agentic-AI-Tutorials/MCP_tutorial/03_MCP_Clients/01_Math_Local_Server/01_math_calc_server.py"
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

async def main():
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()

    named_tools = {tool.name: tool for tool in tools}

    print("Available tools:", list(named_tools.keys()))

    llm = ChatOpenAI(model="gpt-4o")  # Updated to a valid model
    llm_with_tools = llm.bind_tools(tools)

    history = []

    while True:
        user_input = input("Enter your query (q or exit to quit): ")
        if user_input.lower() in ['q', 'exit']:
            break

        history.append(HumanMessage(content=user_input))

        # Agent loop to handle potential multiple rounds of tool calls
        while True:
            # Limit to last 10 messages for context
            input_messages = history[-10:]

            response = await llm_with_tools.ainvoke(input_messages)
            history.append(response)

            if not getattr(response, "tool_calls", None):
                print("AI:", response.content)
                break

            tool_messages = []
            for tc in response.tool_calls:
                selected_tool = named_tools[tc["name"]]
                print(f"Using tool: {selected_tool.name}")
                
                args = tc.get("args") or {}
                result = await selected_tool.ainvoke(args)
                tool_messages.append(ToolMessage(tool_call_id=tc["id"], content=json.dumps(result)))

            history.extend(tool_messages)

if __name__ == '__main__':
    asyncio.run(main())