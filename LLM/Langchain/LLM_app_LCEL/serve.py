from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langserve import add_routes

load_dotenv()

# Create model
groq_api_key = os.getenv("GROQ_API_KEY")
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
model = ChatGroq(model = "llama-3.1-8b-instant",groq_api_key = groq_api_key)

# Define parser
from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()

## Prompt templates using parameters
from langchain_core.prompts import ChatPromptTemplate
generic_template = "Translate the following into {language}"

prompt = ChatPromptTemplate.from_messages(
    [("system",generic_template),
     ("user","{text}")]
)

# Chain elements
chain = prompt|model|parser
chain.invoke({"language":"Italian","text":"Hello, how are you?"})

app = FastAPI(title="Langchain Server", version ="1.0", 
              description="Simple API server using langchain runnable interfaces"
               )

add_routes(app,
           chain,
           path="/chain")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8000)