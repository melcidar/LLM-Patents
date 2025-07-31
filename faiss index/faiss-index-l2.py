import faiss
import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm

# 📁 Putanja do foldera s podacima
data_folder = "E:/Meliha-Patenti/output-joblib"

# 🔍 Učitaj sve fajlove
embedding_files = sorted(glob.glob(os.path.join(data_folder, "embeddings_batch_*.npy")))
metadata_files = sorted(glob.glob(os.path.join(data_folder, "metadata_batch_*.csv")))

print(f"🔍 Pronađeno {len(embedding_files)} .npy fajlova i {len(metadata_files)} .csv fajlova")

# 📐 Dimenzija embeddinga iz prvog fajla
sample = np.load(embedding_files[0])
dimension = sample.shape[1]

# 📦 Učitaj više uzoraka za treniranje FAISS indexa (npr. 1M embeddinga = prvih 20 fajlova)
print("📥 Učitavam više embeddinga za treniranje FAISS indexa (npr. 1M redova)...")
train_embeddings = []

for f in tqdm(embedding_files[:68], desc="📊 Trening podaci"): # 20 za 1 milion podataka, 360 za 80%, 225 za 50%, 100 za oko 20%, 90 za 20%
    train_embeddings.append(np.load(f))

train_sample = np.vstack(train_embeddings).astype("float32")

# 🏗️ Kreiraj IVF index i treniraj ga
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, 1000)  # 1000 klastera
index.train(train_sample)

print(f"✅ FAISS IndexIVFFlat treniran na {train_sample.shape[0]} embeddinga")

# 📋 Učitavanje i dodavanje svih embeddinga i metapodataka u index
all_metadata = []

for emb_file, meta_file in tqdm(zip(embedding_files, metadata_files), total=len(embedding_files), desc="🔗 Indexiranje"):
    emb = np.load(emb_file).astype("float32")
    df = pd.read_csv(meta_file)

    index.add(emb)
    all_metadata.append(df)

# 💾 Snimi FAISS index
index_path = os.path.join(data_folder, "faiss_index_ivf_70batch.bin")
faiss.write_index(index, index_path)

'''
# 💾 Snimi sve metapodatke
metadata = pd.concat(all_metadata, ignore_index=True)
metadata_path = os.path.join(data_folder, "metadata_all.csv")
metadata.to_csv(metadata_path, index=False)

print(f"📄 Metadata CSV snimljen: {metadata_path}")
'''
print(f"\n✅ FAISS index snimljen: {index_path}")

