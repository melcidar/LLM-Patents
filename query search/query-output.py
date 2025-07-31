import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


# === Load index and metadata ===
index = faiss.read_index("E:\\Meliha-Patenti\\index\\faiss_index_ivf_cosine.bin")
metadata = pd.read_csv("E:\\Meliha-Patenti\\metadata_all.csv")

# === Load sentence embedding model ===
model = SentenceTransformer(".\\local_minilm_l6_v2")  # Add device="cuda" if you have GPU

# === Prompt user for query ===
query = input("\n🔍 Enter your patent query:\n> ")

# === Embed query ===
query_embedding = model.encode([query], normalize_embeddings=True)

# === Search FAISS index ===
k = 10
distances, indices = index.search(query_embedding.astype("float32"), k)

# === Retrieve metadata for top-k ===
top_k_metadata = metadata.iloc[indices[0]]

# === Display results ===
print("\n📄 Top-k Results:")
for rank, (row, dist) in enumerate(zip(top_k_metadata.itertuples(index=False), distances[0]), 1):
    print(f"\nRank {rank}")
    print(f"Distance: {dist:.4f}")
    print(f"Patent ID: {row.patent_id}")
    print(f"Claim: {row.claim_text}")
    print("-" * 50)
