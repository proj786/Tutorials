from langchain_openai import chatopenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.header('Research Tool')

user_input =st.text_input('Enter your prompt')

if st.button:
    st.text('Some random text')