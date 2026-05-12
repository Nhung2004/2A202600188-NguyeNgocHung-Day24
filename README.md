# Lab 24 — Full Evaluation & Guardrail System

## Overview
The "Lab 24 — Full Evaluation & Guardrail System" project represents a critical step in transitioning a Proof-of-Concept (PoC) RAG system into a production-ready application. In the modern AI landscape of 2026, building a chatbot that "seems to work" is no longer sufficient; enterprises require rigorous, quantifiable evidence of accuracy, safety, and reliability. This project addresses these requirements through a four-phase implementation of industry-leading evaluation and protection frameworks.

The system is built on a foundation of automated metrics using RAGAS (Retrieval-Augmented Generation Assessment), which evaluates the pipeline across four key dimensions: Faithfulness, Answer Relevancy, Context Precision, and Context Recall. By generating synthetic test sets from the document corpus, we ensure that the evaluation is both objective and scalable. Beyond automated metrics, the project implements a sophisticated "LLM-as-a-Judge" pipeline. This component leverages advanced reasoning models to perform pairwise comparisons between different RAG versions, incorporating robust bias mitigation strategies like "swap-and-average" to ensure that position and length biases do not skew the results.

Safety is handled by a comprehensive Guardrails Stack that protects the system at every stage of interaction. The input layer includes PII (Personally Identifiable Information) redaction using Microsoft Presidio and custom Vietnamese regex patterns, ensuring data privacy compliance. A topic scope validator ensures the model stays within its intended domain, while adversarial testing hardens the system against jailbreak attempts. Finally, the output layer utilizes Llama Guard 3 logic to prevent the generation of unsafe or prohibited content. All components are monitored for latency to ensure that these security layers do not degrade the user experience, aiming for a production SLO of P95 < 2.5 seconds. This holistic approach ensures that the RAG pipeline is not only accurate but also ethically sound and resilient against malicious use.

## Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate # or venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables in a `.env` file:
   ```text
   OPENAI_API_KEY=your_key_here
   ```

## Results Summary

### Phase A (RAGAS)
- **Test set:** 5 questions (Generated from Lab 24 assignment instructions).
- **Metrics:** (Results pending execution).
- **Failure Clusters:** 2 clusters identified (Complex Reasoning, Context Missing).
- See [phase-a/failure_analysis.md](phase-a/failure_analysis.md) for details.

### Phase B (LLM-Judge)
- **Cohen's kappa vs human:** 0.000 (Simulated disagreement).
- **Bias Mitigation:** Swap-and-average successfully handled position bias.
- **Observations:** Position bias was mitigated (0.0% A-win bias). Length bias remains a factor (100% preference for longer answers in small sample). Details in [phase-b/judge_bias_report.md](phase-b/judge_bias_report.md).

### Phase C (Guardrails)
- **PII Detection Rate:** 100% for test identifiers (Email, CCCD, Phone, Tax Code).
- **Topic Validator Accuracy:** 60.0% on mixed query types.
- **Adversarial Defense:** Successfully blocked 80% of off-topic and adversarial prompts.
- **Llama Guard Latency P95:** ~1.0s (Simulated via GPT-4o-mini).

### Phase D (Blueprint)
Detailed production architecture and SLOs are documented in [phase-d/blueprint.md](phase-d/blueprint.md).

## Lessons Learned
- **Evaluation is Iterative:** RAGAS provides a great baseline, but manual failure analysis is crucial to identify specific retrieval vs. generation gaps.
- **Guards have Overhead:** Parallelizing guardrail checks (PII, Topic) is essential to keep end-to-end latency within acceptable SLOs (P95 < 2.5s).
- **Judge Calibration:** Even advanced models like GPT-4o-mini exhibit biases; swap-and-average is a non-negotiable step for reliable automated judging.

## Demo Video
(Link to be provided upon submission)
