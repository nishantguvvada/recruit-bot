from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

def create_vector_db():

    # Generate a vector db out of the csv rows

    # loader = CSVLoader(file_path='./test_data.csv',
    #     csv_args={
    #     'delimiter': ',',
    # })

    loader = PyPDFDirectoryLoader(path='./documents')

    documents = loader.load()

    # Character splitting
    # text_splitter=CharacterTextSplitter(separator = "\n", chunk_size=10, chunk_overlap=1)

    # Text-structure based splitting
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
    # ISSUE WITH SPLITTING TEXT WITHIN A PDF - loses context of the whole document i.e. chunks from a 1 page resume in pdf format do not have any context overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)

    docs=text_splitter.split_documents(documents)

    # embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.getenv('OPENAI_API_KEY'))
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.getenv('GEMINI_API_KEY'))
    # embeddings = MistralAIEmbeddings(
    #     model='mistral-embed',
    #     api_key=os.getenv('MISTRAL_KEY')
    # )

    vectorstore_faiss=FAISS.from_documents(docs, embeddings)
    vectorstore_faiss.save_local("faiss_index")
    print("Index created.")

def vector_db():

    # embeddings = MistralAIEmbeddings(
    #     model='mistral-embed',
    #     api_key=os.getenv('MISTRAL_KEY')
    # )

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.getenv('GEMINI_API_KEY'))

    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    return db
