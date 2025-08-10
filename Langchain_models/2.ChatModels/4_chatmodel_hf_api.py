from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3",
    task = "text-generation"

)
model = ChatHuggingFace(llm = llm )
result = model.invoke("What is capital of India")
print(result.content)