# Failure Cluster Analysis

## Bottom 10 Questions

| # | Question | F | AR | CP | CR | Avg |
|---|---|---|---|---|---|---|
| 1 | What is the purpose of the Blueprint Document in t... | 0.00 | 0.00 | 1.00 | 0.00 | 0.25 |
| 2 | What is the focus of Task A.4 in the evaluation pr... | 0.00 | 0.00 | 0.33 | 1.00 | 0.33 |
| 3 | What is the significance of VinUniversity in the c... | 0.00 | 0.00 | 1.00 | 0.50 | 0.37 |
| 4 | What are the key components and acceptance criteri... | 1.00 | 0.98 | 1.00 | 0.50 | 0.87 |
| 5 | Chào mừng đến với Chương 5, bạn có thể cho biết mụ... | 1.00 | 0.80 | 0.83 | 1.00 | 0.91 |

## Clusters Identified

### Cluster C1: Complex Reasoning Failures
**Pattern:** Questions requiring multi-step inference from the lab guide.
**Root cause:** RAG pipeline retrieval depth is insufficient for complex logic.
**Proposed fix:** Increase top-k and use a more advanced reranker.

### Cluster C2: Context Missing
**Pattern:** Questions about specific technical details not captured in top chunks.
**Root cause:** Chunk size or overlap issues.
**Proposed fix:** Implement semantic chunking or adjust chunk overlap.
