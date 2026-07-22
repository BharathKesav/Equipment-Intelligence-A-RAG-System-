
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from dotenv import load_dotenv
load_dotenv()

print("Importing pipeline...")
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever

query = "What engine model powers the Cat 538?"

# Test dense alone first
print("Testing dense retriever...")
dense = DenseRetriever()
try:
    results = dense.search(query, top_k=20)
    print(f"Dense OK — got {len(results)} results")
except Exception as e:
    print(f"Dense FAILED: {e}")

# Test sparse alone
print("Testing sparse retriever...")
sparse = SparseRetriever()
try:
    results = sparse.search(query, top_k=20)
    print(f"Sparse OK — got {len(results)} results")
except Exception as e:
    print(f"Sparse FAILED: {e}")

print("Both done.")