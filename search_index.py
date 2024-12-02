import faiss
import pandas as pd
import numpy as np
import openai

# Set your OpenAI API key
openai.api_key = "your_openai_api_key"

# Load your CSV file (needed to map search results back to products)
csv_file = "products.csv"
data = pd.read_csv(csv_file)

# Combine product name and description for embeddings
data['content'] = data['product_name'] + " " + data['description']

# Function to generate embeddings
def get_embeddings(texts):
    """
    Generate embeddings using OpenAI's embedding model.
    Replace with your embedding generator if needed.
    """
    embeddings = []
    for text in texts:
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        embeddings.append(response['data'][0]['embedding'])
    return embeddings

# Load the FAISS index
index = faiss.read_index("product_index.faiss")
print("FAISS index loaded successfully.")

# Function to search the index
def search_index(query, top_k=5):
    """
    Search the FAISS index for similar products.
    """
    # Generate embedding for the query
    query_embedding = np.array(
        get_embeddings([query])
    ).astype('float32')

    # Perform search
    distances, indices = index.search(query_embedding, top_k)

    # Fetch results
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx != -1:  # Ensure valid index
            results.append((data.iloc[idx]['product_name'], dist))
    return results

# Example: Query the index
if __name__ == "__main__":
    while True:
        user_query = input("\nSearch for a product (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        print("\nTop matches for your query:")
        results = search_index(user_query)
        for name, dist in results:
            print(f"Product: {name}, Similarity Score: {1 - dist:.2f}")
