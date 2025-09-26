"""
build_index.py
---------------
Builds a FAISS index from the merged tweets dataset.
Stores:
- FAISS index: tweets_index.faiss
- Metadata DataFrame: tweets_metadata.pkl
"""

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

# Paths
DATA_PATH = "../data/tweets_merged.csv"
INDEX_PATH = "tweets_index.faiss"
META_PATH = "tweets_metadata.pkl"

def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    # Keep only necessary columns
    if "celebrity" not in df.columns or "text" not in df.columns:
        raise ValueError("CSV must contain at least ['celebrity','date','text'] columns")

    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Compute embeddings
    print("Generating embeddings...")
    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity (because normalized)
    index.add(np.array(embeddings))

    # Save index + metadata
    print(f"Saving FAISS index to {INDEX_PATH}")
    faiss.write_index(index, INDEX_PATH)

    print(f"Saving metadata to {META_PATH}")
    df.to_pickle(META_PATH)

    print("Index build complete ✅")

if __name__ == "__main__":
    main()
