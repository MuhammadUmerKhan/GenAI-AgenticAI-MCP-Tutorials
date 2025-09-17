from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Initialize models
model = ChatGroq(model="openai/gpt-oss-20b")

prompt1 = PromptTemplate(
    template="Write one line joke on the topic - {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Write one line explanation of the joke - {joke}",
    input_variables=['joke']
)

parser = StrOutputParser()

joke = RunnableSequence(prompt1, model, parser)

# RunnableParallel returns the same value as output as it gets value as input
chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke, chain).invoke({"topic":"AI"})

print(final_chain)