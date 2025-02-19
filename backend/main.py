# from vector import vector_db

# vector_db = vector_db()

# results = vector_db.similarity_search(query="Software Developer",k=3)
# for doc in results:
#     print(f"* {doc.page_content} [{doc.metadata}]")

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders.csv_loader import CSVLoader
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
from uuid import uuid4
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

print(os.getenv('OPENAI_API_KEY'))

# embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.getenv('OPENAI_API_KEY'))

@tool
def add_numbers(a: int, b: int):
    # a tool that fetches data from the vector db (RAG) or any other database (Direct DB call)
    """
    Adds the 2 numbers
    """
    return a + b

llm = OpenAI(model_name='gpt-3.5-turbo-instruct', api_key=os.getenv('OPENAI_API_KEY'))
tools = [add_numbers]
agent_executor = create_react_agent(llm, tools)


# print(agent_executor.invoke({"input": "What is 4+5?"}))


