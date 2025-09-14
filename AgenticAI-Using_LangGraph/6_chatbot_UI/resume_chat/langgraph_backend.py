from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

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

# Initialize checkpointer
checkpointer = MemorySaver()

# Compile graph with checkpointer
chatbot = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    while True:
        user_input = input("👤 Type Here: ")
        if user_input.strip().lower() in ['exit', 'quit', 'bye']:
            print("👋 Ending chat...")
            break
        
        response = chatbot.invoke(
            {'messages': [HumanMessage(content=user_input)]},
            config={'configurable': {'thread_id': 'default'}}
        )
        bot_reply = response['messages'][-1].content
        print(f"🤖 Bot: {bot_reply}")