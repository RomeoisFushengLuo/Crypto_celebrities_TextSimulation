import argparse
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import openai   # or any LLM client you use

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))

from retrieve import retrieve_tweets
# --------------------------
# Load resources (once)
# --------------------------
print("Loading index, metadata, and embedding model...")
index = faiss.read_index("embeddings/tweets_index.faiss")
df = pd.read_pickle("embeddings/tweets_metadata.pkl")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------
# Retrieval function
# --------------------------
def retrieve_tweets(celebrity, event_desc, top_k=5):
    sub_df = df[df["celebrity"] == celebrity].reset_index(drop=True)
    if sub_df.empty:
        raise ValueError(f"No tweets found for {celebrity}")

    q_emb = embed_model.encode([event_desc], normalize_embeddings=True)

    # per-celebrity embeddings
    sub_embs = embed_model.encode(sub_df["text"].tolist(), normalize_embeddings=True)

    # build temp FAISS index
    sub_index = faiss.IndexFlatIP(q_emb.shape[1])
    sub_index.add(sub_embs)

    sims, idxs = sub_index.search(q_emb, top_k)
    return sub_df.iloc[idxs[0]][["date", "text"]]

# --------------------------
# Prompt builder
# --------------------------
def build_prompt(celebrity, event, top_k=10):
    examples = retrieve_tweets(celebrity, event, top_k)
    tweets_sample = "\n".join(
        f"- ({row.date.strftime('%Y-%m-%d')}): {row.text}"
        for _, row in examples.iterrows()
    )
    user_prompt = f"""
Celebrity: {celebrity}
Event: {event}

Here are some of their past tweets to guide the style:
---
{tweets_sample}
---

Task:
- Generate 2-3 plausible new tweets in this persona’s style
- Limit to 280 characters each
- Reflect how this persona might react to the event
"""
    return user_prompt

# --------------------------
# Call the LLM
# --------------------------
def generate_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-5-2025-09-25",   # replace with your model
        messages=[
            {"role": "system", "content": "You simulate persona-style tweets."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=300,
    )
    return response["choices"][0]["message"]["content"]

# --------------------------
# Main script
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--celebrity", type=str, required=True)
    parser.add_argument("--event", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    print(f"Simulating response for {args.celebrity} to event: {args.event}")
    prompt = build_prompt(args.celebrity, args.event, top_k=args.top_k)
    output = generate_response(prompt)

    print("\n=== Prompt Sent to LLM ===")
    print(prompt)
    print("\n=== Simulated Response ===")
    print(output)
