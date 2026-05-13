# How We Built a Production-Ready RAG Evaluation & Guardrail System

In the world of Generative AI, building a RAG (Retrieval-Augmented Generation) system that "works" is only the first 20%. The remaining 80% is about ensuring it is accurate, safe, and reliable. In Lab 24, we tackled this challenge head-on by building a comprehensive evaluation and protection stack.

## 1. Automated Evaluation with RAGAS
We started by generating a synthetic test set of 50 questions directly from our document corpus. Using **RAGAS**, we measured four critical metrics:
- **Faithfulness:** Does the answer stick to the facts?
- **Answer Relevancy:** Does it actually answer the user's question?
- **Context Precision & Recall:** How good is our retrieval?

Our results showed that while our baseline was decent, multi-hop reasoning remained a challenge—leading us to implement a reranker for better context quality.

## 2. LLM-as-a-Judge: Scaling Human Judgment
To move beyond simple metrics, we built an **LLM-as-a-Judge** pipeline. To avoid biases (like position bias and length bias), we implemented:
- **Swap-and-average:** Running each comparison twice with orders swapped.
- **Cross-judge aggregation:** Using two different models (GPT-4o-mini and GPT-3.5-Turbo) to find consensus.

This human-calibrated pipeline (measured via Cohen's Kappa) gives us confidence that our automated evaluations mirror human judgment.

## 3. The Guardrails Stack: Safety First
Accuracy doesn't matter if your bot leaks PII or gets jailbroken. Our stack includes:
- **PII Redaction:** A hybrid layer of Microsoft Presidio and custom Vietnamese regex.
- **Topic Scope Validator:** Ensuring the bot stays on-topic using embedding similarity.
- **Output Guardrails:** Leveraging Llama Guard 3 logic to prevent unsafe responses.

## Key Takeaway
Evaluation is not a one-time task; it's a continuous loop. By building this system, we've moved from "guessing" if our AI is good to "knowing" exactly where it stands.

---
*Created as part of Lab 24 - Full Evaluation & Guardrail System.*
