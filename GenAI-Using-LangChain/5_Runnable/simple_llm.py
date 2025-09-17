from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()   

model = ChatGroq(model="openai/gpt-oss-20b")

topic = input("Enter a topic: ")

prompt = PromptTemplate(
    template='Generate a 5 lines report on {topic}',
    input_variables=['topic']
)

result = model.invoke(prompt.format(topic=topic)).content

print(result)