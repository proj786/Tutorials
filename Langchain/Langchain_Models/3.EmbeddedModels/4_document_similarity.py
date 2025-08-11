from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model = 'text-embedding-3-large',dimensions=300)

documemts =[ "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."]

query = "Tell me about Virat Kohli"

doc_embeddings = embedding.embed_documents(documemts)
query_embeddings = embedding.embed_query(query)
#all the parameter should be 2-D list
scores = cosine_similarity([query_embeddings],doc_embeddings)[0]

print(list(enumerate(scores)))
index ,score =  sorted(list(enumerate(scores)),key = lambda x:x[1])[-1]
print(query)
print(documemts[index])
print('similarity score is :',score)