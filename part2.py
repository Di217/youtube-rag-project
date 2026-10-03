import os
import warnings
import chromadb

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

warnings.filterwarnings(
    "ignore",
    message=".*unauthenticated requests to the HF Hub.*"
)
from youtube_rag import load_transcript
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
def spiliter(url):
    text = load_transcript(url)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_text(text)
    print(f"Total chunks: {len(chunks)}")
    return chunks
def generate_embeddings(chunks):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode( chunks, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True )
    print("Embeddings shape:", embeddings.shape)
    return embeddings 

def store_chunks(chunks, embeddings):

    # Step 1: Connect to your existing ChromaDB database
    client = chromadb.PersistentClient(path="./chroma_db")

    # Step 2: Access your existing collection
    collection = client.get_or_create_collection(
        name="youtube_transcripts",
        configuration={"hnsw": {"space": "cosine"}}
    )

    # Step 3: Delete old chunks and embeddings, if any exist
    if collection.count() > 0:
        old_ids = collection.get()["ids"]
        collection.delete(ids=old_ids)
        print("Old chunks and embeddings deleted.")

    # Step 4: Store the new chunks and embeddings
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist()
    )

    # Step 5: Display the number of stored chunks
    print("Stored chunks:", collection.count())