from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

template1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='Generate a 5 line summary on the following report: \n {report}',
    input_variables=['report']
)

# With output parser

parser = StrOutputParser()

chain = template1 | model | parser | model | template2 | model | parser

result = chain.invoke({"topic": "Black Hole"})

print(result)


# Without output parser

# result1 = model.invoke(prompt1.format(topic="Artificial Intelligence"))
# print(result1.content)

# result2 = model.invoke(prompt2.format(report=result1.content))

# print(result2.content)