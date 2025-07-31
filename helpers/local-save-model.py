from sentence_transformers import SentenceTransformer

# This will download and cache it locally
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.save("local_minilm_l6_v2")
