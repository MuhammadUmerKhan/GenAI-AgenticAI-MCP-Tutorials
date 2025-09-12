from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import os, sqlite3

_ = load_dotenv()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
llm = ChatGroq(model="openai/gpt-oss-20b")

def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Create state graph
graph = StateGraph(ChatState)

# Add nodes
graph.add_node("chat_node", chat_node)

# Add edges
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# Create database file if it doesn't exist
os.makedirs("database", exist_ok=True)
if not os.path.exists(os.path.join("database", "chat_history.db")):
    conn = sqlite3.connect(os.path.join("database", "chat_history.db"), check_same_thread=False)
else:
    conn = sqlite3.connect(os.path.join("database", "chat_history.db"), check_same_thread=False)

# Initialize checkpointer
checkpointer = SqliteSaver(conn)

# Compile graph with checkpointer
chatbot = graph.compile(checkpointer=checkpointer)

def get_all_threads():
    all_threads = set()

    for chk in checkpointer.list(None):
        all_threads.add(chk.config['configurable']['thread_id'])
        return list(all_threads)

# if __name__ == "__main__":
#     while True:
#         user_input = input("👤 Type Here: ")
#         if user_input.strip().lower() in ['exit', 'quit', 'bye']:
#             print("👋 Ending chat...")
#             break
        
#         response = chatbot.invoke(
#             {'messages': [HumanMessage(content=user_input)]},
#             config={'configurable': {'thread_id': 'default'}}
#         )
#         bot_reply = response['messages'][-1].content
#         print(f"🤖 Bot: {bot_reply}")