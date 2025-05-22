# ingest.py
from sentence_transformers import SentenceTransformer
import whisper
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


# Initialize models
whisper_model = whisper.load_model("base")  # or "medium", "large" depending on system resources
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Set up Chroma DB
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="chroma_db"))
collection = client.get_or_create_collection(name="meeting_notes",
                                             embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(embedding_model))

def transcribe_and_ingest(video_path, meeting_id):
    print(f"Transcribing {video_path}...")
    result = whisper_model.transcribe(video_path)
    transcription = result["text"]

    print("Splitting transcription...")
    chunks = [transcription[i:i+500] for i in range(0, len(transcription), 500)]
    ids = [f"{meeting_id}_{i}" for i in range(len(chunks))]

    print("Storing embeddings in Chroma DB...")
    collection.add(documents=chunks, ids=ids, metadatas=[{"meeting_id": meeting_id}] * len(chunks))
    
    return transcription  # return full transcription for PDF or future use
