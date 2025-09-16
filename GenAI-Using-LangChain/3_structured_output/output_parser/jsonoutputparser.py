from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

# using jsonoutputparser you can enfore schema to ouput

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me name and age of a person: \n {format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({})

print(result)