from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableBranch, RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Initialize models
model = ChatGroq(model="openai/gpt-oss-20b")

prompt1 = PromptTemplate(
    template="Write a detailed report on the {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Summarize this report - {report}",
    input_variables=['report']
)

parser = StrOutputParser()

report = prompt1 | model | parser # -> LCEL: LangChain Expression Lnaguage which introduce by LnagChain because Sequence runnable are uses everywhere in common so replace the RunnableSequence to | (pipe) operator
summarize = RunnableSequence(prompt2, model, parser)

chain_branch_chain = RunnableBranch(
    (lambda x:len(x.split())>500, summarize),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report, chain_branch_chain)
print(final_chain.invoke({"topic":"AI"}))