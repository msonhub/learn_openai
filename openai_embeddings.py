import os
from openai import OpenAI
import numpy as np

def cosine_similarity(v1, v2):
    return np.dot(v1, v2)

client = OpenAI(
  api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
  base_url="https://mrs-test-1.openai.azure.com/openai/v1/"
)

try:

    response = client.embeddings.create(
        input = "The quick brown fox jumped over the lazy dog.",
        model= "text-embedding-ada-002"
    )

    #print(response.model_dump_json(indent=2))

    embedding1 = response.data[0].embedding

    #print(f"Embedding vector length: {len(embedding)}")

    print(embedding1[:20])

    response = client.embeddings.create(
        input = "Wild animals jump over domesticated animals.",
        model= "text-embedding-ada-002"
    )

    embedding2 = response.data[0].embedding

    print(embedding2[:20])

    response = client.embeddings.create(
        input = "Is fox a domestic animal?",
        model= "text-embedding-ada-002"
    ) 

    query = response.data[0].embedding

    similarities = [cosine_similarity(query, facts) for facts in [embedding1, embedding2]] 

    print(similarities)

    most_similar_index = np.argmax(similarities)

    print(f"Most similar index={most_similar_index}") 
except Exception as e:
    print(f"An error occured: {e}")


