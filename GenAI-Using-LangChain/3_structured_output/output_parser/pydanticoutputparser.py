from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person", gt=18)
    city: str = Field(description="City of the person")
    
parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Give me name and age of a person of {place}: \n {format_instruction}",
    input_variables=['place'],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({"place": "Pakistan"})

print(result)