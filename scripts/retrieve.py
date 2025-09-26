"""
retrieve.py
------------
Provides functions to retrieve tweets similar in meaning to an event description,
filtered by celebrity.
"""

import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# Paths
INDEX_PATH = "../embeddings/tweets_index.faiss"
META_PATH = "../embeddings/tweets_metadata.pkl"

# Load resources
print("Loading index and metadata...")
index = faiss.read_index(INDEX_PATH)
df = pd.read_pickle(META_PATH)

print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_tweets(celebrity: str, event_desc: str, top_k: int = 5):
    """
    Retrieve top-k tweets for a celebrity most relevant to an event description.
    Args:
        celebrity (str): celebrity name as in the dataset
        event_desc (str): description of the crypto event
        top_k (int): number of tweets to retrieve
    Returns:
        pd.DataFrame: top-k retrieved tweets with ['date','text']
    """
    # Filter tweets by celebrity
    sub_df = df[df["celebrity"] == celebrity].reset_index(drop=True)
    if sub_df.empty:
        raise ValueError(f"No tweets found for {celebrity}")

    # Embed the event description
    q_emb = embed_model.encode([event_desc], normalize_embeddings=True)

    # Build temp FAISS index per celebrity
    sub_embs = embed_model.encode(sub_df["text"].tolist(), normalize_embeddings=True)
    sub_index = faiss.IndexFlatIP(q_emb.shape[1])
    sub_index.add(sub_embs)

    sims, idxs = sub_index.search(q_emb, top_k)

    return sub_df.iloc[idxs[0]][["date", "text"]]

if __name__ == "__main__":
    # quick test
    results = retrieve_tweets("APompliano", "Bitcoin hits new all-time high", top_k=3)
    print(results)
