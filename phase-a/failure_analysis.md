# Failure Cluster Analysis

## Bottom 10 Questions

| # | Question | F | AR | CP | CR | Avg |
|---|---|---|---|---|---|---|
| 1 | What are the key components and tasks involved in ... | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | What does Task A.3 involve in Lab 24?... | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 | What is the focus of Task A.4 in Lab 24?... | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | What does AICB stand for in the context of the lab... | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5 | VinUniversity la gi?... | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | How can Loom be utilized in the demo video for Lab... | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 7 | VinUniveristy la gi?... | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 8 | How can Loom be utilized in the demo video for Lab... | 0.00 | 0.00 | 0.00 | 0.20 | 0.05 |
| 9 | What skills will a Data Science Trainee learn by 2... | 0.00 | 0.00 | 0.33 | 0.00 | 0.08 |
| 10 | What are the key considerations for using AICB in ... | 0.00 | 0.00 | 0.00 | 0.50 | 0.12 |

## Clusters Identified

### Cluster C1: Complex Reasoning Failures
**Pattern:** Questions requiring multi-step inference from the lab guide.
**Root cause:** RAG pipeline retrieval depth is insufficient for complex logic.
**Proposed fix:** Increase top-k and use a more advanced reranker.

### Cluster C2: Context Missing
**Pattern:** Questions about specific technical details not captured in top chunks.
**Root cause:** Chunk size or overlap issues.
**Proposed fix:** Implement semantic chunking or adjust chunk overlap.
