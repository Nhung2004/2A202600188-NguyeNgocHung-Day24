# Implementation Plan - Lab 24: Full Evaluation & Guardrail System

This plan outlines the steps to build a production-ready evaluation and guardrail system for the RAG pipeline.

## User Review Required

> [!IMPORTANT]
> - **API Keys:** I have the OpenAI key. I may need a **Groq API Key** for Llama Guard 3 (Task C.4) if we want to follow the recommended free-tier path. If not provided, I will attempt to use OpenAI as a fallback judge for safety.
> - **Document Corpus:** I extracted `data/corpus.md` from the lab PDF. I will use this as the primary corpus for evaluation tasks.
> - **Human Calibration (Task B.3):** This requires manual labeling. I will simulate the human labels for demonstration purposes, but you may want to review them.
> - **Disk Space:** C: drive is full (0.00 GB). I am saving files to D: drive for now.

## Proposed Changes

### Phase A: RAGAS Evaluation (30 points)

#### [NEW] `scripts/phase_a_eval.py`
- **Task A.1:** Generate a synthetic test set of 50 questions from `data/corpus.md` (50% simple, 25% reasoning, 25% multi-context).
- **Task A.2:** Run RAGAS metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall).
- **Task A.3:** Perform Failure Cluster Analysis on the bottom 10 questions.
- **Task A.4:** Create `.github/workflows/eval-gate.yml`.

### Phase B: LLM-as-Judge & Calibration (25 points)

#### [NEW] `scripts/phase_b_judge.py`
- **Task B.1:** Implement Pairwise Judge with swap-and-average bias mitigation.
- **Task B.2:** Implement Absolute Scoring with a 4-point rubric.
- **Task B.3:** Calculate Cohen's Kappa comparing LLM Judge vs simulated human labels.
- **Task B.4:** Generate Bias Observation Report.

### Phase C: Guardrails Stack (35 points)

#### [NEW] `scripts/phase_c_guardrails.py`
- **Task C.1:** Input Guardrail: PII Redaction (Presidio + VN Regex).
- **Task C.2:** Input Guardrail: Topic Scope Validator (Embedding similarity).
- **Task C.3:** Adversarial Testing (DAN, jailbreaks, etc.).
- **Task C.4:** Output Guardrail: Llama Guard 3 (via Groq or fallback).
- **Task C.5:** Full Stack Integration and Latency Benchmarking.

### Phase D: Blueprint Document (10 points)

#### [NEW] `phase-d/blueprint.md`
- SLO definitions, Architecture diagram (Mermaid), Alert playbook, and Cost Analysis.

---

## Verification Plan

### Automated Tests
- Run `scripts/phase_a_eval.py` and verify `ragas_summary.json` and `testset_v1.csv` are created.
- Run `scripts/phase_b_judge.py` and verify `pairwise_results.csv` and `kappa` calculation.
- Run `scripts/phase_c_guardrails.py` and verify `pii_test_results.csv` and `latency_benchmark.csv`.

### Manual Verification
- Review the generated `blueprint.md`.
- Review the `judge_bias_report.md`.
