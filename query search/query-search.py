import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from datetime import datetime

# Učitaj FAISS index i metadata
index = faiss.read_index("faiss_index.bin")
metadata = pd.read_csv("metadata_all.csv")

# Učitaj model
model = SentenceTransformer("local_minilm_l6_v2")  # ili npr. "BAAI/bge-small-en"

def search_and_save(query, top_k=5):
    # Embeduj query
    query_embedding = model.encode([query], normalize_embeddings=False)

    # Pretraži u FAISS
    distances, indices = index.search(query_embedding, top_k)

    # Pripremi rezultate
    results = []
    for i, idx in enumerate(indices[0]):
        row = {
            "rank": i + 1,
            "query": query,
            "patent_id": metadata.loc[idx, "patent_id"],
            "claim": metadata.loc[idx, "cleaned_claim"],
            "distance": distances[0][i]
        }
        results.append(row)
        # Prikaz u terminalu
        print(f"\n🔹 Result #{i + 1}")
        print(f"Patent ID: {row['patent_id']}")
        print(f"Claim: {row['claim']}")
        print(f"Distance: {row['distance']:.4f}")

    # Sačuvaj u CSV
    df_results = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"query_results_{timestamp}.csv"
    df_results.to_csv(filename, index=False)
    print(f"\n📁 Rezultati su sačuvani u fajl: {filename}")

if __name__ == "__main__":
    while True:
        query = input("\n🔎 Unesi svoj query (ili 'exit' za izlaz): ").strip()
        if query.lower() == "exit":
            break
        search_and_save(query)
