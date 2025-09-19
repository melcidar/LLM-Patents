import psycopg2
from pgvector.psycopg2 import register_vector

TABLE = "public.patent_claims_loadtest"   # <-- change to whatever you want
DIM   = 3                                  # <-- set your real embedding dim later (e.g., 384)

conn = psycopg2.connect("postgresql://postgres:fablabSQL@192.168.50.70:5432/patentdb")
conn.autocommit = True
register_vector(conn)

with conn.cursor() as cur:
    # sanity: where am i?
    cur.execute("""
        SELECT inet_server_addr(), inet_server_port(), current_database(), current_user, current_schema()
    """)
    print("WHERE AM I:", cur.fetchone())

    # ensure vector + table
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(f"CREATE TABLE IF NOT EXISTS {TABLE} (patent_id text, cleaned_claim text, embedding vector({DIM}));")

    # insert 2 test rows
    cur.execute(f"""
        INSERT INTO {TABLE} (patent_id, cleaned_claim, embedding)
        VALUES ('TEST-1','abc','[0.1,0.2,0.3]'),
               ('TEST-2','def','[0.4,0.5,0.6]');
    """)

    # verify
    cur.execute(f"SELECT COUNT(*) FROM {TABLE};")
    print("ROWS NOW:", cur.fetchone()[0])
