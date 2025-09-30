# Crypto Celebrities Text Simulation

This repository simulates how well-known crypto personas might respond to market events.  
It works by:
1. Merging historical tweets from multiple celebrities.  
2. Embedding tweets with a transformer model (`sentence-transformers`).  
3. Storing embeddings in a FAISS index.  
4. Retrieving style-guided examples per celebrity.  
5. Building a prompt to send to an LLM (e.g., OpenAI GPT, Hugging Face models).  

---

## Correspondence between input names (celebrities' tweets account) and their actual names, the user should input the names in the first column.
---- **Author_username** ----  **Author_name** ----
---- APompliano ---- Anthony Pompliano ----
---- CathieDWood ----Cathie Wood ----
---- HesterPeirce ---- Hester Peirce ----
---- Lagarde ---- Christine Lagarde ----
---- Nouriel ---- Nouriel Roubini ----
---- SenWarren_partial / ewarren ---- Elizabeth Warren ----
---- VitalikButerin ---- vitalik.eth ----
---- balajis ---- Balaji Srinivasan ----
---- brian_armstrong ---- Brian Armstrong ----
---- cz_binance_partial ---- Changpeng Zhao ----
---- dtrump ---- Donald Trump ----
---- elonmusk_partial ---- Elon Musk ----
---- saylor ---- Michael Saylor ----
---- steve_hanke ---- Steve Hanke ---- 





## 📂 Repository Structure

Crypto_celebrities_TextSimulation/
│
├── data/
│ └── tweets_merged.csv # merged dataset of tweets (celebrity, date, text, etc.)
├── embeddings/
│ ├── build_index.py # script: build FAISS index from dataset
│ ├── tweets_index.faiss # FAISS index (stored via Git LFS)
│ └── tweets_metadata.pkl # tweet metadata (stored via Git LFS)
├── scripts/
│ └── retrieve.py # retrieval functions (search similar tweets)
├── prompts/
│ └── simulate_reaction.yaml # LLM prompt template
└── run_simulation.py # main entry point


---

## ⚙️ Setup

### 1. Clone Repo with LFS Support
Since large files are tracked with **Git LFS**, make sure you have it installed:

```bash
brew install git-lfs    # macOS
git lfs install

### Then clone:
git clone https://github.com/RomeoisFushengLuo/Crypto_celebrities_TextSimulation.git
cd Crypto_celebrities_TextSimulation

### This will pull both code and the large FAISS/metadata files.


## Usage
### Step 1. Build Index (optional)

### If you modify the dataset (data/tweets_merged.csv), rebuild the FAISS index:
---
cd embeddings
python build_index.py
---

### This regenerates:

### tweets_index.faiss

### tweets_metadata.pkl
### Step 2. Retrieve Similar Tweets

### You can test retrieval manually:
---
cd scripts
python retrieve.py
---



#### Example:
---
from retrieve import retrieve_tweets
results = retrieve_tweets("APompliano", "Bitcoin hits new all-time high", top_k=3)
print(results)
---

###Step 3. Run Full Simulation

###From the repo root:
---
python run_simulation.py --celebrity "APompliano" --event "Bitcoin crashes 20% overnight"
---

This will:
Retrieve top-k relevant tweets by the celebrity.

Insert them into the YAML prompt (prompts/simulate_reaction.yaml).

Send the prompt to the chosen LLM backend.

Print simulated tweet-style responses.
