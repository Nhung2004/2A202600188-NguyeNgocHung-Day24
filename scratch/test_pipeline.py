import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import build_pipeline, run_query

load_dotenv()

def test_pipeline():
    print("Building pipeline...")
    search, reranker = build_pipeline()
    print("Running test query...")
    query = "What is the main objective of Lab 24?"
    answer, contexts = run_query(query, search, reranker)
    print(f"\nQuery: {query}")
    print(f"Answer: {answer}")
    print(f"Contexts found: {len(contexts)}")

if __name__ == "__main__":
    test_pipeline()
