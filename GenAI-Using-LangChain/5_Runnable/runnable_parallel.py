from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Initialize models
model1 = ChatGroq(model="openai/gpt-oss-20b")
model2 = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

# Define prompts
prompt1 = PromptTemplate(
    template="Write simple short tweet on the topic - {topic1}",
    input_variables=["topic1"]
)

prompt2 = PromptTemplate(
    template="Write simple and short linked post on the topic - {topic2}",
    input_variables=['topic2']
)

parser = StrOutputParser()

# Define the parallel chain with RunnableSequence objects
chain_parallel = RunnableParallel({
    "tweet": RunnableSequence(prompt1, model1, parser),
    "linkedIn": RunnableSequence(prompt2, model2, parser)
})

# Invoke the parallel chain with the input
response = chain_parallel.invoke({"topic1": "AI", "topic2":"Robotics"})
print(response)