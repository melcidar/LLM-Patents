import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer
from multiprocessing import Pool, cpu_count

# Load environment variables
load_dotenv()

# DB config
server = os.getenv("DB_SERVER")
database = os.getenv("DB_NAME")
connection_string = (
    'mssql+pyodbc:///?odbc_connect='
    f'DRIVER={{ODBC Driver 18 for SQL Server}};'
    f'SERVER={server};'
    f'DATABASE={database};'
    'Trusted_Connection=yes;'
    'TrustServerCertificate=yes;'
)
engine = create_engine(connection_string)

# Text cleaning
def clean_claim(text):
    if not text:
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'the present invention.*?\.', '', text, flags=re.IGNORECASE)
    return text.strip()

# Chunk processor — runs in each process
def process_chunk(chunk):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    model = SentenceTransformer("local_bge_model")  # local download to avoid HF rate limits
    chunk['cleaned_claim'] = chunk['claim_text'].apply(clean_claim)
    embeddings = model.encode(chunk['cleaned_claim'].tolist(), batch_size=256, show_progress_bar=False)
    return (chunk[['patent_id', 'cleaned_claim']], embeddings)

# Main
def main():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    os.makedirs("output", exist_ok=True)
    batch_size = 100_000  # adjust based on memory capacity
    offset = 0
    batch_num = 0
    max_workers = min(cpu_count(), 12)

    with engine.connect() as conn:
        while True:
            query = f"""
                SELECT patent_id, claim_text
                FROM independent_claims_sample
                ORDER BY patent_id
                OFFSET {offset} ROWS
                FETCH NEXT {batch_size} ROWS ONLY
            """
            df = pd.read_sql(query, conn)

            if df.empty:
                print("✅ Done! No more rows.")
                break

            chunks = np.array_split(df, max_workers)
            print(f"🔄 Processing batch {batch_num} — Rows: {len(df)} using {max_workers} workers...")

            with Pool(processes=max_workers) as pool:
                results = list(tqdm(pool.imap(process_chunk, chunks), total=len(chunks), desc="🧠 Embedding chunks"))

            meta_parts, embed_parts = zip(*results)
            meta_df = pd.concat(meta_parts, ignore_index=True)
            embed_matrix = np.vstack(embed_parts)

            meta_df.to_csv(f"output/metadata_batch_{batch_num}.csv", index=False)
            np.save(f"output/embeddings_batch_{batch_num}.npy", embed_matrix)

            print(f"✅ Saved batch {batch_num} — Rows: {len(meta_df)}")

            offset += batch_size
            batch_num += 1

if __name__ == "__main__":
    main()
