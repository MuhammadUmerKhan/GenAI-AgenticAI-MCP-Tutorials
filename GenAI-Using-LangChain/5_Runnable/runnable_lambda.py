from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda
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

def word_counter(text):
    return len(text.split())

# RunnableLambda is used to convert any custom python function into a Runnable
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'count': RunnableLambda(word_counter)
})

chain = RunnableSequence(joke, parallel_chain)

print(chain.invoke({"topic":"AI"}))