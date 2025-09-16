from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda

load_dotenv()

class Sentiment(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Return sentiment of the review either negative or positive")

model = ChatGroq(model="openai/gpt-oss-20b")

parser1 = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=Sentiment)

prompt = PromptTemplate(
    template="Classify the sentiment into positive or negative \n {text} \n {format_instructions}",
    input_variables=["text"],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)

classifier_chain = prompt | model | parser2

# print(classifier_chain.invoke({"text": "I like this movie"}).sentiment)

prompt1 = PromptTemplate(
    template="Write a appropriate response to this positve feedback: {feedback}",
    input_variables=["feedback"]
)

prompt2 = PromptTemplate(
    template="Write a appropriate response to this negative feedback: {feedback}",
    input_variables=["feedback"]
)

positive_chain = prompt1 | model | parser1
negative_chain = prompt2 | model | parser1

branch_chain  = RunnableBranch(
    (lambda x: x.sentiment == "positive", positive_chain),
    (lambda x: x.sentiment == "negative", negative_chain),
    RunnableLambda(lambda x: None)
)

chain = classifier_chain | branch_chain

result = chain.invoke({"text": "I like this movie"})

print(result)

chain.get_graph().print_ascii()