import os
from dotenv import load_dotenv
import os
from vector import vector_db
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq

load_dotenv()

faiss_index = vector_db()

retriever = faiss_index.as_retriever(search_kwargs={"k": 1})

vector_search = create_retriever_tool(
    retriever=retriever,
    name="search_candidates",
    description="Search and retrieve top candidates with the relevant skill sets.",
)

tools = [vector_search]

llm = ChatMistralAI(model_name='mistral-small-latest', api_key=os.getenv('MISTRAL_KEY'))

agent = create_react_agent(
        llm,
        tools,
        state_modifier=(
            "You are a recruitment analysis expert that can analyse candidate information"
            "You MUST only respond in 100 words."
            "You MUST include human-readable response before transferring to another agent."
        ),
    )

response = agent.invoke({"messages": "What is the salary expection of an IT Manager?"})

print("Response", response)

# embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.getenv('OPENAI_API_KEY'))

# @tool
# def add_numbers(a: int, b: int):
#     # a tool that fetches data from the vector db (RAG) or any other database (Direct DB call)
#     """
#     Adds the 2 numbers
#     """
#     return a + b

# llm = OpenAI(model_name='gpt-3.5-turbo-instruct', api_key=os.getenv('OPENAI_API_KEY'))
# tools = [add_numbers]
# agent_executor = create_react_agent(llm, tools)


# print(agent_executor.invoke({"input": "What is 4+5?"}))


