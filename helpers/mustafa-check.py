import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

server = os.getenv("DB_SERVER")
database = os.getenv("DB_NAME")

# Setup engine
connection_string = (
    'mssql+pyodbc:///?odbc_connect='
    f'DRIVER={{ODBC Driver 18 for SQL Server}};'
    f'SERVER={server};'
    f'DATABASE={database};'
    'Trusted_Connection=yes;'
    'TrustServerCertificate=yes;'
)
engine = create_engine(connection_string)

# Your two lists of patent numbers
list_a = ['8805552', '9678522', '10394268', '10396592']
list_b = ['9563215', '10429871', '11126213', '11782470', '12181904',
          '9979198', '9880580', '8862279', '9639103', '10261536',
          '11561564', '11747849']

# Convert lists to SQL-safe format
list_a_sql = ",".join([f"'{p}'" for p in list_a])
list_b_sql = ",".join([f"'{p}'" for p in list_b])

# Get all claims for both sets
query = f"""
SELECT patent_id, claim_text
FROM independent_claims
WHERE patent_id IN ({list_a_sql}) OR patent_id IN ({list_b_sql})
"""

df = pd.read_sql(text(query), engine)

# Split claims into two groups
claims_a = df[df['patent_id'].isin(list_a)]
claims_b = df[df['patent_id'].isin(list_b)]

# Compare claims: is any claim text from A in claims of B?
matches = []

for idx_a, row_a in claims_a.iterrows():
    for idx_b, row_b in claims_b.iterrows():
        if row_a['claim_text'].strip() == row_b['claim_text'].strip():
            matches.append({
                "patent_a": row_a['patent_id'],
                "patent_b": row_b['patent_id'],
                "matched_claim": row_a['claim_text']
            })

# Display results
if matches:
    result_df = pd.DataFrame(matches)
    print("✅ Matching claims found:")
    print(result_df)
else:
    print("❌ No matching claims found.")
