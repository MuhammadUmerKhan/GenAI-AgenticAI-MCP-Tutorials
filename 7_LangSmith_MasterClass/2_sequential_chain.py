from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

# different project\
os.environ["LANGCHAIN_PROJECT"] = "sequential_chain_demo"

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

config = {
    'run_name': 'sequential_chain_demo',
    'tags': ['qa', 'sequential', 'summarization'],
    'metadata': {
        'model': 'meta-llama/llama-4-scout-17b-16e-instruct'
    }
}

result = chain.invoke({'topic': 'Unemployment in World'}, config=config)

print(result)
