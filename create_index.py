import faiss
import pandas as pd
import numpy as np
import openai

# Set your OpenAI API key
openai.api_key = "your_openai_api_key"

# Load your CSV file
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

# Generate embeddings for all products
print("Generating embeddings...")
embeddings = get_embeddings(data['content'].tolist())

# Convert to NumPy array
embedding_matrix = np.array(embeddings).astype('float32')

# Create FAISS index
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)  # Use L2 distance for similarity
index.add(embedding_matrix)  # Add embeddings to the index
print(f"Index created with {index.ntotal} vectors.")

# Save the index for later use
faiss.write_index(index, "product_index.faiss")
print("Index saved to 'product_index.faiss'.")
