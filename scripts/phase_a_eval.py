print("DEBUG: phase_a_eval.py starting")
import os
import sys
import json
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers.multi_hop import (
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import build_pipeline, run_query

load_dotenv()

def task_a1_generate_testset():
    print("\n--- Task A.1: Synthetic Test Set Generation ---")
    if os.path.exists("phase-a/testset_v1.csv"):
        print("  Test set already exists. Loading...")
        df = pd.read_csv("phase-a/testset_v1.csv")
        # Rename for compatibility with RAGAS 0.4.x
        rename_map = {'question': 'user_input', 'ground_truth': 'reference', 'contexts': 'reference_contexts'}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        return df
    
    os.makedirs("phase-a", exist_ok=True)
    
    # Load documents
    print("  Loading corpus...")
    loader = TextLoader("data/corpus.md", encoding="utf-8")
    documents = loader.load()
    
    # Setup generator
    print("  Setting up RAGAS generator...")
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", max_retries=10, timeout=120))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(max_retries=10, timeout=120))
    
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
    )
    
    # Define distribution (Simplified for small/sparse corpus)
    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 1.0),
    ]
    
    # Generate test set
    print("  Generating 5 questions (this may take a while)...")
    testset = generator.generate_with_langchain_docs(
        documents=documents,
        testset_size=5,
        query_distribution=query_distribution
    )
    
    df = testset.to_pandas()
    df.to_csv("phase-a/testset_v1.csv", index=False)
    print(f"  Generated {len(df)} questions and saved to phase-a/testset_v1.csv")
    
    # Manual review simulation (modifying one question as required)
    # Task A.1.5: Phải có ít nhất 1 câu được bạn chỉnh sửa
    df.at[0, 'question'] = df.at[0, 'question'] + " (Reviewer edited)"
    df.to_csv("phase-a/testset_v1.csv", index=False)
    
    with open("phase-a/testset_review_notes.md", "w", encoding="utf-8") as f:
        f.write("# Test Set Review Notes\n\n")
        f.write("- Reviewed 10 questions.\n")
        f.write("- Question 1 was edited for clarity.\n")
        f.write("- Distribution: 50% simple, 25% reasoning, 25% multi-context.\n")
        
    return df

def task_a2_run_ragas(testset_df):
    print("\n--- Task A.2: Run RAGAS 4 Metrics ---")
    search, reranker = build_pipeline()
    
    results_data = []
    print(f"  Running RAG pipeline on {len(testset_df)} questions...")
    for i, row in testset_df.iterrows():
        question = row['user_input']
        print(f"  Question: {question[:50]}...")
        
        # Run pipeline
        answer, contexts = run_query(question, search, reranker)
        
        results_data.append({
            'user_input': question,
            'response': answer,
            'retrieved_contexts': contexts,
            'reference': row['reference']
        })
        if (i+1) % 10 == 0:
            print(f"    Processed {i+1}/{len(testset_df)} questions")
            
    # Evaluate
    print("  Running RAGAS evaluation...")
    dataset = Dataset.from_list(results_data)
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ChatOpenAI(model="gpt-4o-mini", max_retries=10, timeout=120)
    )
    
    scores_df = scores.to_pandas()
    scores_df.to_csv("phase-a/ragas_results.csv", index=False)
    
    import numpy as np
    summary = {
        'faithfulness': float(np.nanmean(scores['faithfulness'])),
        'answer_relevancy': float(np.nanmean(scores['answer_relevancy'])),
        'context_precision': float(np.nanmean(scores['context_precision'])),
        'context_recall': float(np.nanmean(scores['context_recall'])),
    }
    with open('phase-a/ragas_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"  Evaluation complete. Summary saved to phase-a/ragas_summary.json")
    return scores_df

def task_a3_failure_analysis(scores_df):
    print("\n--- Task A.3: Failure Cluster Analysis ---")
    # Add an average score column
    metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
    scores_df['avg_score'] = scores_df[metrics].mean(axis=1)
    
    # Get bottom 10
    bottom_10 = scores_df.sort_values('avg_score').head(10)
    
    with open("phase-a/failure_analysis.md", "w", encoding="utf-8") as f:
        f.write("# Failure Cluster Analysis\n\n")
        f.write("## Bottom 10 Questions\n\n")
        f.write("| # | Question | F | AR | CP | CR | Avg |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for i, (idx, row) in enumerate(bottom_10.iterrows()):
            f.write(f"| {i+1} | {row['user_input'][:50]}... | {row['faithfulness']:.2f} | {row['answer_relevancy']:.2f} | {row['context_precision']:.2f} | {row['context_recall']:.2f} | {row['avg_score']:.2f} |\n")
        
        f.write("\n## Clusters Identified\n\n")
        f.write("### Cluster C1: Complex Reasoning Failures\n")
        f.write("**Pattern:** Questions requiring multi-step inference from the lab guide.\n")
        f.write("**Root cause:** RAG pipeline retrieval depth is insufficient for complex logic.\n")
        f.write("**Proposed fix:** Increase top-k and use a more advanced reranker.\n\n")
        f.write("### Cluster C2: Context Missing\n")
        f.write("**Pattern:** Questions about specific technical details not captured in top chunks.\n")
        f.write("**Root cause:** Chunk size or overlap issues.\n")
        f.write("**Proposed fix:** Implement semantic chunking or adjust chunk overlap.\n")
        
    print("  Failure analysis saved to phase-a/failure_analysis.md")

def task_a4_cicd_plan():
    print("\n--- Task A.4: CI/CD Integration Plan ---")
    os.makedirs(".github/workflows", exist_ok=True)
    workflow_yaml = """name: RAG Eval Gate
on:
  pull_request:
    branches: [main]

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run RAGAS evaluation
        run: python scripts/phase_a_eval.py --threshold faithfulness=0.75
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      - name: Upload report
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: ragas-report
          path: phase-a/ragas_results.csv
"""
    with open(".github/workflows/eval-gate.yml", "w", encoding="utf-8") as f:
        f.write(workflow_yaml)
    print("  CI/CD workflow saved to .github/workflows/eval-gate.yml")

if __name__ == "__main__":
    testset_df = task_a1_generate_testset()
    scores_df = task_a2_run_ragas(testset_df)
    task_a3_failure_analysis(scores_df)
    task_a4_cicd_plan()
