from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()   

model = ChatGroq(model="openai/gpt-oss-20b")

prompt = PromptTemplate(
    template="Write one line joke about the topic {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

chain = RunnableSequence(prompt, model, parser)

joke = chain.invoke({"topic":"AI"})
print(joke)