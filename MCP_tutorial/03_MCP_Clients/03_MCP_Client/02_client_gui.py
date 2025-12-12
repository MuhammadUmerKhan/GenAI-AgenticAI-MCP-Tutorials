# app.py — Clean & Robust MCP + Streamlit Chat with Proper Async & History Management

import json, asyncio
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

load_dotenv()

# ==================== CONFIG: MCP Servers ====================
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

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "When using tools, do not say things like 'Let me calculate' or 'Fetching data'. "
    "Just call the tool when needed, then give a concise final answer."
)

# ==================== Streamlit UI Setup ====================
st.set_page_config(page_title="MCP Tool Chat", page_icon="Tools", layout="centered")
st.title("Tools MCP-Powered Assistant")
st.caption("Math • Expense Tracker • Manim Animation • Powered by Local + Remote MCP Servers")

# ==================== Initialize Session State ====================
if "initialized" not in st.session_state:
    with st.spinner("Starting MCP servers and loading tools..."):
        # Run async init in a proper event loop
        client = MultiServerMCPClient(SERVERS)
        tools = asyncio.run(client.get_tools())

        tool_by_name = {t.name: t for t in tools}

        llm = ChatOpenAI(model="gpt-4o", temperature=0)  # gpt-5 → use existing model
        llm_with_tools = llm.bind_tools(tools)

        st.session_state.update({
            "client": client,
            "tools": tools,
            "tool_by_name": tool_by_name,
            "llm": llm,
            "llm_with_tools": llm_with_tools,
            "history": [SystemMessage(content=SYSTEM_PROMPT)],
            "initialized": True,
        })

# ==================== Helper: Keep only last N messages ====================
def trim_history(messages, max_messages=20):
    """Keep system prompt + last N messages"""
    system = [m for m in messages if isinstance(m, SystemMessage)]
    others = [m for m in messages if not isinstance(m, SystemMessage)]
    return system + others[-max_messages:]

# ==================== Render Chat History ====================
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").markdown(msg.content)
    elif isinstance(msg, AIMessage):
        # Only show final AI responses, skip intermediate ones with tool_calls
        if not getattr(msg, "tool_calls", None):
            st.chat_message("assistant").markdown(msg.content or "")
    # ToolMessages are hidden from UI

# ==================== User Input ====================
if prompt := st.chat_input("Ask me anything... (e.g., draw a rotating cube, calculate 123! + 456, track expense)"):
    # Add user message
    st.chat_message("user").markdown(prompt)
    st.session_state.history.append(HumanMessage(content=prompt))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # First: Let LLM decide if tools are needed
            response = asyncio.run(
                st.session_state.llm_with_tools.ainvoke(st.session_state.history)
            )

            # Append the (possibly tool-calling) assistant message
            st.session_state.history.append(response)

            if not getattr(response, "tool_calls", None):
                # Direct answer
                st.markdown(response.content or "")
            else:
                # Execute all tool calls
                tool_messages = []
                placeholder = st.empty()

                for tc in response.tool_calls:
                    tool_name = tc["name"]
                    args = tc.get("args") or {}

                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except:
                            pass  # keep as string if invalid JSON

                    tool = st.session_state.tool_by_name[tool_name]
                    result = asyncio.run(tool.ainvoke(args))
                    tool_messages.append(
                        ToolMessage(tool_call_id=tc["id"], content=json.dumps(result, indent=2))
                    )

                    # Optional: show live tool output (great for debugging or transparency)
                    with placeholder.container():
                        st.code(f"Tool: {tool_name}\nInput: {json.dumps(args, indent=2)}\nOutput: {json.dumps(result, indent=2)}", language="json")

                st.session_state.history.extend(tool_messages)

                # Final response after tools
                final_response = asyncio.run(
                    st.session_state.llm.ainvoke(st.session_state.history)
                )

                st.markdown(final_response.content or "")
                st.session_state.history.append(AIMessage(content=final_response.content or ""))

    # Trim history to prevent token blowup
    st.session_state.history = trim_history(st.session_state.history, max_messages=20)

    # Force rerun to update chat
    st.rerun()