import asyncio, json
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

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
    }
}

async def main():
    client = MultiServerMCPClient(SERVERS)
    client_tools = await client.get_tools()
    # print(client_tools)
    
    tools = {}
    for tool in client_tools:
        tools[tool.name] = tool
    
    # print(tools)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm_with_tools = llm.bind_tools(client_tools)
    
    prompt = "What is the product of 10 and 15"
    response = await llm_with_tools.ainvoke(prompt)
    
    selected_tools = response.tool_calls[0]['name']
    selected_tool_args = response.tool_calls[0]['args']
    
    print("Selected tool:", selected_tools)
    print("Selected tool args:", selected_tool_args)

    
if __name__ == "__main__":
    asyncio.run(main())