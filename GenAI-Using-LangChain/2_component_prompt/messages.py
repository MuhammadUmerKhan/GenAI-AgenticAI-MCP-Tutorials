from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What's 1 + 1?")
]

model = ChatGroq(model="openai/gpt-oss-20b")

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))

print(messages)