import os
import numpy as np, pandas as pd, psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

TABLE_NAME   = "patent_claims"
BATCH_START  = 0
BATCH_END    = 449
META_TMPL    = "E:\Meliha Kasapovic - ETSI and AI\Patents embeddings and metadata\output-joblib\metadata_batch_{i}.csv"
EMB_TMPL     = "E:\Meliha Kasapovic - ETSI and AI\Patents embeddings and metadata\output-joblib\embeddings_batch_{i}.npy"
PAGE_SIZE    = 1000

def fmeta(i): return META_TMPL.format(i=i)
def femb(i):  return EMB_TMPL.format(i=i)

conn = psycopg2.connect("postgresql://postgres:fablabSQL@192.168.50.70:5432/patentdb")
conn.autocommit = True
conn.set_client_encoding('UTF8')
with conn.cursor() as cur:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
register_vector(conn)

def table_count():
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};")
        return cur.fetchone()[0]

pre_count = table_count()

insert_sql = f"INSERT INTO {TABLE_NAME} (patent_id, cleaned_claim, embedding) VALUES %s"

expected_total = 0   # how many rows we *intend* to insert (sum of batch sizes)
inserted_total = 0   # how many rows the DB reports as inserted

for i in range(BATCH_START, BATCH_END + 1):
    meta_path, emb_path = fmeta(i), femb(i)

    if not (os.path.exists(meta_path) and os.path.exists(emb_path)):
        print(f"[{i}] Skipping (missing files): {meta_path} / {emb_path}")
        continue

    try:
        meta = pd.read_csv(meta_path, usecols=["patent_id", "cleaned_claim"])
        X = np.load(emb_path, mmap_mode="r").astype("float32")

        if len(meta) != X.shape[0]:
            print(f"[{i}] Row mismatch (meta={len(meta)} vs emb={X.shape[0]}). Skipping.")
            continue

        rows = [(pid, txt, emb.tolist())
                for pid, txt, emb in zip(meta["patent_id"], meta["cleaned_claim"], X)]

        expected_total += len(rows)

        with conn.cursor() as cur:
            execute_values(cur, insert_sql, rows, page_size=PAGE_SIZE)
            # rowcount should equal number of inserted rows for this statement
            inserted = cur.rowcount if cur.rowcount is not None else len(rows)
            inserted_total += inserted

        print(f"[{i}] Insert attempted: {len(rows)} | reported inserted: {inserted}")

    except Exception as e:
        print(f"[{i}] FAILED: {e}")

post_count = table_count()
delta = post_count - pre_count

print("\n=== LOAD AUDIT ===")
print(f"Expected to insert: {expected_total}")
print(f"DB reported inserted (sum of rowcount): {inserted_total}")
print(f"Table rows before: {pre_count}")
print(f"Table rows after : {post_count}")
print(f"Observed delta   : {delta}")

if delta != inserted_total:
    print("WARNING: observed table delta != reported inserted. Investigate logs/batches above.")
elif inserted_total != expected_total:
    print("NOTE: reported inserted != expected (did you skip batches or have mismatches?).")
else:
    print("All rows accounted for.")
