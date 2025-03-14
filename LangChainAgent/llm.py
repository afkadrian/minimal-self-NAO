import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
# Initialize the LLM
gpt4 = ChatOpenAI(model="gpt-4o", api_key=os.getenv('OPENAI_API_KEY'))
llama = ChatOllama(model="llama3.1")