# AI Prompts Used in Lab 24

## Phase A: RAGAS Evaluation
- **Test Set Generation:** Used RAGAS `TestsetGenerator` with default evolutionary prompts (SingleHop).
- **Evaluation:** Used RAGAS default metrics prompts for Faithfulness, Answer Relevancy, Context Precision, and Context Recall.

## Phase B: LLM-as-Judge
- **Pairwise Judge Prompt:**
```text
You are an impartial evaluator. Compare two answers to the same question.
Question: {question}
Answer A: {answer_a}
Answer B: {answer_b}
Rate based on:
- Factual accuracy
- Relevance to question
- Conciseness
Output JSON only:
{"winner": "A" or "B" or "tie", "reason": "..."}
```

- **Absolute Scoring Prompt:**
```text
Score the answer on 4 dimensions, each 1-5 scale:
1. Factual accuracy (1=many errors, 5=fully accurate)
2. Relevance (1=off-topic, 5=directly answers)
3. Conciseness (1=verbose, 5=appropriately brief)
4. Helpfulness (1=unclear, 5=actionable)
Question: {question}
Answer: {answer}
Output JSON only:
{"accuracy": int, "relevance": int, "conciseness": int, "helpfulness": int, "overall": float}
```

## Phase C: Guardrails
- **Topic Scope Prompt:**
```text
Is this question about one of these topics: {allowed_topics}?
Question: {text}
Answer YES or NO only.
```
