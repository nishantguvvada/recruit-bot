import os
from dotenv import load_dotenv
from vector import vector_db, create_vector_db
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
# from langchain_mistralai import ChatMistralAI
# from langchain_groq import ChatGroq
# from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def generate_response(user_query: str):

    # Initialize vector database
    faiss_index = vector_db()

    # create_vector_db() # Uncomment to create the vector index

    # Setup a retriever
    retriever = faiss_index.as_retriever(search_kwargs={"k": 20})

    # Create a retriever tool
    vector_search = create_retriever_tool(
        retriever=retriever,
        name="search_candidates_database",
        description="Provide candidates information.",
    )

    # List of tools
    tools = [vector_search]

    # Models
    # llm = ChatMistralAI(model_name='mistral-small-latest', api_key=os.getenv('MISTRAL_KEY'))
    # llm = ChatGroq(model='mixtral-8x7b-32768', api_key=os.getenv('GROQ_KEY'))
    # llm = ChatAnthropic(model_name='claude-3-5-sonnet-latest', max_tokens=1024, max_retries=2, api_key=os.getenv('CLAUDE_KEY'))
    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', api_key=os.getenv('GEMINI_API_KEY'))

    # Bind llm with tools into a React Agent
    agent = create_react_agent(
        llm,
        tools,
        state_modifier=(
            "You are a recruitment analysis expert that can analyse candidate information"
            "You MUST only respond in 100 words."
        )
    )

    response = agent.invoke({"messages": user_query})

    return response["messages"][-1].content


