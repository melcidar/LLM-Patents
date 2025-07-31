import pandas as pd
import re

# Load your base DataFrame
df_base = pd.read_csv("check1_with_retrieved_cos.csv")

# List of your 3 Innography result CSVs, in the same order as rows in df_base
innography_files = [
    "test 3 (1).csv",
    "test 2 (1).csv",
    "test 1 (1).csv"
]

# Function to extract the numeric part from a publication number
def extract_number(pub):
    match = re.match(r'^[A-Z]{2}(\d+)', str(pub))
    return match.group(1) if match else None

# Process each Innography result file
top100_all = []

for file in innography_files:
    df_inno = pd.read_csv(file, encoding='ISO-8859-1')  # or 'latin1'

    # Extract cleaned patent numbers
    numbers = df_inno['Publication Number'].dropna().apply(extract_number)
    numbers = numbers.dropna().unique()[:100]  # Keep only first 100 unique
    top100_string = " | ".join(numbers)
    top100_all.append(top100_string)

# Fill in the new column in your base DataFrame
df_base['top 100 retrieved innography patents'] = top100_all

total_found = []

for _, row in df_base.iterrows():
    # Get top 10 and top 100 as sets of patent IDs
    top_10 = set(str(row['top 10 retrieved patents']).split(" | ")) if pd.notna(row['top 10 retrieved patents']) else set()
    top_100 = set(str(row['top 100 retrieved innography patents']).split(" | ")) if pd.notna(row['top 100 retrieved innography patents']) else set()
    
    # Remove empty string artifacts
    top_10.discard('')
    top_100.discard('')

    # Count how many in top_10 exist in top_100
    intersection_count = len(top_10.intersection(top_100))
    total_found.append(intersection_count)

# Assign the results
df_base['total found'] = total_found

# Save the updated file
df_base.to_csv("check1_final.csv", index=False)

print("✅ 'total found' column added and saved to 'check1_final.csv'.")

