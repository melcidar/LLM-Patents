import faiss
import numpy as np
import pandas as pd
import glob

# Učitaj sve embeddinge i metapodatke
embedding_files = sorted(glob.glob("output-joblib/embeddings_batch_*.npy"))
metadata_files = sorted(glob.glob("output-joblib/metadata_batch_*.csv"))

all_embeddings = []
all_metadata = []

for emb_file, meta_file in zip(embedding_files, metadata_files):
    all_embeddings.append(np.load(emb_file))
    all_metadata.append(pd.read_csv(meta_file))

X = np.vstack(all_embeddings)
metadata = pd.concat(all_metadata, ignore_index=True)

# FAISS index
dimension = X.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(X)

faiss.write_index(index, "faiss_index.bin")
metadata.to_csv("metadata_all.csv", index=False)
print("✅ FAISS index and metadata saved!")
