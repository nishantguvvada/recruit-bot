from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

def create_vector_db():

    # Function to generate a vector db out of the csv rows

    loader = CSVLoader(file_path='./test_data.csv',
        csv_args={
        'delimiter': ',',
    })
    documents = loader.load()

    # Character splitting
    text_splitter=CharacterTextSplitter(separator = "\n", chunk_size=10, chunk_overlap=1)
    
    docs=text_splitter.split_documents(documents)

    print("Subscripting docs[0]: ", docs[0])

    print("OPENAI_KEY: ", os.getenv('OPENAI_API_KEY'))

    # embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.getenv('OPENAI_API_KEY'))
    embeddings = MistralAIEmbeddings(
        model='mistral-embed',
        api_key=os.getenv('MISTRAL_KEY')
    )

    vectorstore_faiss=FAISS.from_documents(docs, embeddings)
    vectorstore_faiss.save_local("faiss_index")

def vector_db():

    embeddings = MistralAIEmbeddings(
        model='mistral-embed',
        api_key=os.getenv('MISTRAL_KEY')
    )

    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    return db