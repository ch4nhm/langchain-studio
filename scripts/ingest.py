import os
import re
import time
import uuid
import hashlib
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import tiktoken
import chromadb

# To resolve import path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../backend"))
from app.core.config import settings

def count_tokens(text: str) -> int:
    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    return len(enc.encode(text))

def clean_text(text: str) -> str:
    """Remove sensitive info (e.g., email, phone numbers)"""
    # Remove simple email patterns
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL REDACTED]', text)
    # Remove phone number patterns (basic)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE REDACTED]', text)
    return text

def get_file_hash(filepath: str) -> str:
    hasher = hashlib.md5()
    with open(filepath, 'rb') as afile:
        buf = afile.read()
        hasher.update(buf)
    return hasher.hexdigest()

def ingest_docs(data_dir: str = "../data"):
    print(f"Starting ingestion from {data_dir}...")
    start_time = time.time()
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("Data directory created. Please put markdown files in it.")
        return

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=settings.OPENAI_API_KEY)
    client = chromadb.PersistentClient(path=os.path.join("../backend", settings.CHROMA_PERSIST_DIRECTORY))
    
    vectorstore = Chroma(
        client=client,
        collection_name="tech_docs",
        embedding_function=embeddings
    )
    
    # Simple idempotency tracking
    processed_files_record = os.path.join(data_dir, ".processed_files.txt")
    processed_hashes = set()
    if os.path.exists(processed_files_record):
        with open(processed_files_record, "r", encoding="utf-8") as f:
            for line in f:
                processed_hashes.add(line.strip())
                
    new_hashes = set()

    docs_to_insert = []
    total_tokens = 0
    
    for filename in os.listdir(data_dir):
        if not filename.endswith(".md"):
            continue
            
        filepath = os.path.join(data_dir, filename)
        file_hash = get_file_hash(filepath)
        
        if file_hash in processed_hashes:
            continue # Skip already processed
            
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()
            
        cleaned_text = clean_text(raw_text)
        
        doc = Document(page_content=cleaned_text, metadata={"source": filename, "file_hash": file_hash})
        docs_to_insert.append(doc)
        new_hashes.add(file_hash)
        
    if not docs_to_insert:
        print("No new documents to ingest.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = splitter.split_documents(docs_to_insert)
    
    # Assign UUID to each chunk and preserve source context
    ids = []
    for i, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid4())
        ids.append(chunk_id)
        chunk.metadata["chunk_id"] = chunk_id
        chunk.metadata["chunk_index"] = i
        # Calculate tokens
        total_tokens += count_tokens(chunk.page_content)
        
    print(f"Inserting {len(chunks)} chunks into vector store...")
    vectorstore.add_documents(documents=chunks, ids=ids)
    
    # Update idempotency record
    with open(processed_files_record, "a", encoding="utf-8") as f:
        for h in new_hashes:
            f.write(h + "\n")
            
    elapsed = time.time() - start_time
    print(f"Ingestion completed in {elapsed:.2f} seconds.")
    print(f"Total tokens processed: {total_tokens}")

if __name__ == "__main__":
    ingest_docs(os.path.join(os.path.dirname(__file__), "../data"))
