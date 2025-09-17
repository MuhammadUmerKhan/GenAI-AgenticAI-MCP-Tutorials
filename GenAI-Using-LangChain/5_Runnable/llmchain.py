from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from dotenv import load_dotenv

load_dotenv()   

model = ChatGroq(model="openai/gpt-oss-20b")

prompt = PromptTemplate(
    template='Generate a 5 lines report on {topic}',
    input_variables=['topic']
)

topic = input("Enter a topic: ")

llm_chain = LLMChain(llm=model, prompt=prompt)
result = llm_chain.invoke(topic)

print(result)