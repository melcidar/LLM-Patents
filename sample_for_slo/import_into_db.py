import numpy as np, pandas as pd, psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

# files
meta = pd.read_csv("metadata_batch_sample_1.csv", usecols=["patent_id", "cleaned_claim"])
X = np.load("embeddings_batch_sample_1.npy", mmap_mode="r").astype("float32")
assert len(meta) == X.shape[0]

rows = [(pid, txt, emb.tolist())
        for pid, txt, emb in zip(meta["patent_id"], meta["cleaned_claim"], X)]

# DB
conn = psycopg2.connect("postgresql://postgres:fablabSQL@192.168.50.70:5432/patentdb")
conn.autocommit = True
conn.set_client_encoding('UTF8')
with conn.cursor() as cur:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")  # makes the 'vector' type in *patentdb*
from pgvector.psycopg2 import register_vector
register_vector(conn)  # will now succeed


with conn, conn.cursor() as cur:
    cur.execute("SELECT current_database(), (SELECT COUNT(*) FROM pg_type WHERE typname='vector')")
    print(cur.fetchone())  # should be ('patentdb', 1)

register_vector(conn)

with conn, conn.cursor() as cur:
    execute_values(cur,
        "INSERT INTO patent_claims_10k_sample (patent_id, cleaned_claim, embedding) VALUES %s",
        rows,
        page_size=1000
    )

print("Imported.")
