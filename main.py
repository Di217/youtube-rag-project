import chromadb
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from google import genai
import os
import warnings
from langchain_core.runnables import RunnableLambda, RunnableParallel
from part2 import spiliter, generate_embeddings, store_chunks
from retriver import retrieve_chunks
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

warnings.filterwarnings(
    "ignore",
    message=".*unauthenticated requests to the HF Hub.*"
)

load_dotenv(dotenv_path=r"C:\Users\divya\OneDrive\Desktop\Langchain\.env", override=True)

api_key = os.getenv("geminikey")
client = genai.Client(api_key=api_key)
if __name__ == "__main__":
    if client is None:
        raise ValueError("Gemini API key not found. Set the 'geminikey' environment variable.")
    main_chain = (
    RunnableLambda(spiliter)
    | RunnableParallel(
        chunks=RunnableLambda(lambda chunks: chunks),
        embeddings=RunnableLambda(generate_embeddings)
    )
    | RunnableLambda(
        lambda data: store_chunks(data["chunks"], data["embeddings"])
    )
    )

    # Take the YouTube link from the user
    link = input("Enter YouTube video link: ").strip()

    # Run the complete chain
    result = main_chain.invoke(link)

    print("Video processed successfully!")
    
    # Take the first question from the user
    question = input("Ask a question about the video: ").strip()

    # Continue until the user says "no"
    while question.lower() != "no":

        # Retrieve the 3 most relevant chunks from ChromaDB
        chunks_found = retrieve_chunks(question, top_k=3)

        # Combine the retrieved chunks into context
        context_text = "\n\n".join(chunks_found)

        # Create the Gemini prompt
        prompt = f"""
        You are a helpful assistant answering questions about a video.

        Use the transcript context below to answer the user's question.
        If the answer is not present in the context, say that you
        cannot find the answer in the provided transcript.
        Do not invent information.

        Transcript context:
        {context_text}

        User question:
        {question}

        Answer:
        """

        # Send the prompt to Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        # Display the answer
        print("\nAnswer:", response.text)

        # Ask whether the user has another question
        question = input(
            "\nIs there any more question you want to ask? "
            "Enter your next question, or type 'no' to stop: "
        ).strip()