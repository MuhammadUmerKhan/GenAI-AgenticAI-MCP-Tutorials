from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b", max_tokens=200)

messages = [
    SystemMessage(content="You are a helpful assistant.")
]

chat_history = []

while True:
    user_input = input("User: ")
    chat_history.append(HumanMessage(content=user_input))
    
    if user_input.lower() in ["exit", "quit", "bye"]:
        break
    result = model.invoke(chat_history)
    
    chat_history.append(AIMessage(content=result.content))
    print("AI:", result.content)

print(chat_history)