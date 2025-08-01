import faiss
import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

data_folder = os.getenv("OUTPUT_DIR")
index_dir = os.getenv("INDEX_DIR")


embedding_files = sorted(glob.glob(os.path.join(data_folder, "embeddings_batch_*.npy")))
metadata_files = sorted(glob.glob(os.path.join(data_folder, "metadata_batch_*.csv")))

sample = np.load(embedding_files[0])
dimension = sample.shape[1]

print("Loading more embeddings to train FAISS index...")
train_embeddings = []

for f in tqdm(embedding_files[:68], desc="Training data:"): # Loaded 68 batches fpr training data
    train_embeddings.append(np.load(f))

train_sample = np.vstack(train_embeddings).astype("float32")


quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, 1000) 
index.train(train_sample)

print(f"FAISS IndexIVFFlat trained on {train_sample.shape[0]} embeddinga")

all_metadata = []

for emb_file, meta_file in tqdm(zip(embedding_files, metadata_files), total=len(embedding_files), desc="🔗 Indexiranje"):
    emb = np.load(emb_file).astype("float32")
    df = pd.read_csv(meta_file)

    index.add(emb)
    all_metadata.append(df)

index_path = os.path.join(data_folder, "faiss_index_ivf_70batch.bin")
faiss.write_index(index, index_path)

'''
# This part was done once and then reused
metadata = pd.concat(all_metadata, ignore_index=True)
metadata_path = os.path.join(data_folder, "metadata_all.csv")
metadata.to_csv(metadata_path, index=False)

print(f"Metadata CSV saved: {metadata_path}")
'''
print(f"\nFAISS index saved: {index_path}")

