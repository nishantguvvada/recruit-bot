from langchain_community.document_loaders.csv_loader import CSVLoader
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import pandas as pd
from uuid import uuid4

load_dotenv()


def vector_db():

    # Function to generate a vector db out of the csv rows

    # loader = CSVLoader(file_path='./resume_data.csv',
    #     csv_args={
    #     'delimiter': ',',
    # })
    # data = loader.load()

    # print(data)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.getenv('OPENAI_API_KEY'))

    # Read CSV file
    df = pd.read_csv('resume_data.csv')

    print(df.head(2))

    uuids = [str(uuid4()) for _ in range(len(df))]

    # Initialize FAISS
    faiss_index = FAISS(
        embedding_function=embeddings,
        index=uuids,
        docstore= InMemoryDocstore(),
        index_to_docstore_id={}
    )

    # Process each row
    for index, row in df.iterrows():
        text = row[index]
        embeddings = embeddings.embed_documents([text])
        faiss_index.from_documents([text], embeddings)

    print("index created!")

    return faiss_index