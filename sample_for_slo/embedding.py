import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer
from joblib import Parallel, delayed, cpu_count
from contextlib import contextmanager

load_dotenv()

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

@contextmanager
def tqdm_joblib(tqdm_object):
    from joblib.parallel import BatchCompletionCallBack
    old_callback = BatchCompletionCallBack.__call__

    def new_callback(self, *args, **kwargs):
        tqdm_object.update(n=self.batch_size)
        return old_callback(self, *args, **kwargs)

    BatchCompletionCallBack.__call__ = new_callback
    try:
        yield tqdm_object
    finally:
        BatchCompletionCallBack.__call__ = old_callback
        tqdm_object.close()

def clean_claim(text):
    if not text:
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'the present invention.*?\.', '', text, flags=re.IGNORECASE)
    return text.strip()

def process_chunk(chunk):
    model = SentenceTransformer("..\local_minilm_l6_v2")
    chunk['cleaned_claim'] = chunk['claim_text'].apply(clean_claim)
    embeddings = model.encode(chunk['cleaned_claim'].tolist(), batch_size=256, show_progress_bar=False)
    return chunk[['patent_id', 'cleaned_claim']], embeddings

def main():
    output_dir = os.getenv("OUTPUT_DIR")
    os.makedirs(output_dir, exist_ok=True)
    batch_size = 50000
    offset = 0
    batch_num = 1

    with engine.connect() as conn:
        while True:
            query = f"""
                SELECT patent_id, claim_text
                FROM claims_sample_10k
                ORDER BY patent_id
                OFFSET {offset} ROWS
                FETCH NEXT {batch_size} ROWS ONLY
            """
            df = pd.read_sql(query, conn)

            if df.empty:
                print("Done! No more rows.")
                break

            num_chunks = min(cpu_count(), 12)
            chunks = np.array_split(df, num_chunks)

            print(f"Processing batch {batch_num} using {num_chunks} jobs...")

            with tqdm_joblib(tqdm(desc="Embedding chunks", total=len(chunks))) as progress_bar:
                results = Parallel(n_jobs=num_chunks)(
                    delayed(process_chunk)(chunk) for chunk in chunks
                )

            meta_parts, embed_parts = zip(*results)
            meta_df = pd.concat(meta_parts, ignore_index=True)
            embed_matrix = np.vstack(embed_parts)

            meta_df.to_csv(f"{output_dir}/metadata_batch_sample_{batch_num}.csv", index=False)
            np.save(f"{output_dir}/embeddings_batch_sample_{batch_num}.npy", embed_matrix)

            print(f"Saved batch {batch_num} — Rows: {len(meta_df)}")

            offset += batch_size
            batch_num += 1

if __name__ == "__main__":
    main()
