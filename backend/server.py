from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from model import generate_response
import os

app = FastAPI()

origins = [
    os.getenv('FRONTEND_URL')
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class UserInput(BaseModel):
    user_input: str

@app.get("/")
def home():
    return {"data": "working"}

@app.post("/response")
def response(query: UserInput):
    agent_response = generate_response(query.user_input)
    return {"query": query.user_input, "response": agent_response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)