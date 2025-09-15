from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

st.header('Reasearch Tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

# template
prompt = load_prompt("prompt.json")

if st.button('Submit'):
    chain = prompt | model
    result = chain.invoke({"paper_input": paper_input, 
                        "style_input": style_input, 
                        "length_input": length_input
                    })
    st.write(result.content)