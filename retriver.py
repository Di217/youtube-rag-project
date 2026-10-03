
import chromadb
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from google import genai
import os
import warnings
from part2 import spiliter, generate_embeddings, store_chunks
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

warnings.filterwarnings(
    "ignore",
    message=".*unauthenticated requests to the HF Hub.*"
)

load_dotenv(dotenv_path=r"C:\Users\divya\OneDrive\Desktop\Langchain\.env", override=True)

api_key = os.getenv("geminikey")
# STEP 1: Load the same embedding model used in part2.py
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# STEP 2: Connect to your EXISTING ChromaDB database
# This path works when you run the script from youtube_ass/
client = chromadb.PersistentClient(path="./chroma_db")


# STEP 3: Access your existing collection
# Do not create a new collection or add the chunks again
collection = client.get_collection(name="youtube_transcripts")


# STEP 4: Define the retriever function
def retrieve_chunks(query, top_k=3):

    # Convert the user's question into a 384-dimensional embedding
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    # Find the most similar stored embeddings in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, collection.count())
    )

    # Extract the text chunks returned by ChromaDB
    retrieved_chunks = results["documents"][0]

    # Return the retrieved text chunks
    return retrieved_chunks


