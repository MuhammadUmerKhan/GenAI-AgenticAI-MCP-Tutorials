from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b", max_tokens=200)

chat_template = ChatPromptTemplate([
    ('system', "You are a helpful {domain} assistant."),
    ('human', "Explain in simple terms, what is: {question}?")
])

prompt = chat_template.invoke({"domain": "mathematics", "question": "algebra"})

result = model.invoke(prompt)
print(result.content)