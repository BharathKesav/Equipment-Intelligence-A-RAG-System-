# test_pipeline.py

# test_pipeline.py — top of file
import os
# --- Add these 3 lines to prevent Windows thread crashes ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Your existing environment variables
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from dotenv import load_dotenv
load_dotenv()
# ... rest of file unchanged

from dotenv import load_dotenv
load_dotenv()

from src.pipeline import run_pipeline
import json

# Test 1 — a question that should be in your documents
print("\n" + "="*60)
print("TEST 1 — In-scope question")
print("="*60)
result = run_pipeline("What engine model powers the Cat 538/538 LL forest machine?")
print(f"\nAnswer:\n{result['answer']}")
print(f"\nCitations used:")
for c in result["citations"]:
    print(f"  [{c['number']}] {c['paragraph_ref']}")
print(f"\nConfidence: {result['confidence']:.3f}")
print(f"Declined:   {result['declined']}")

# Test 2 — a question that should NOT be in your documents
print("\n" + "="*60)
print("TEST 2 — Out-of-scope question (should decline)")
print("="*60)
result2 = run_pipeline("What is the capital of France?")
print(f"\nAnswer:\n{result2['answer']}")
print(f"Declined: {result2['declined']}")
