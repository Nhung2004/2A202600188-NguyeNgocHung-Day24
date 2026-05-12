from phase_c.input_guard import InputGuard
from phase_c.output_guard import OutputGuard
import time

class RAGGuardPipeline:
    def __init__(self, allowed_topics):
        self.input_guard = InputGuard()
        self.output_guard = OutputGuard()
        self.allowed_topics = allowed_topics

    def run(self, user_query, rag_func):
        # 1. Input Guard
        sanitized_query, is_on_topic, in_latency = self.input_guard.sanitize(user_query, self.allowed_topics)
        
        if not is_on_topic:
            return "Sorry, I can only answer questions related to the lab topics.", in_latency
        
        # 2. RAG Execution
        start_rag = time.perf_counter()
        raw_response, _ = rag_func(sanitized_query)
        rag_latency = (time.perf_counter() - start_rag) * 1000
        
        # 3. Output Guard
        is_safe, reason = self.output_guard.validate(raw_response)
        
        if not is_safe:
            return f"Blocked: {reason}", in_latency + rag_latency
        
        return raw_response, in_latency + rag_latency

if __name__ == "__main__":
    # Example usage
    def dummy_rag(q): return f"Answer for {q}", []
    pipeline = RAGGuardPipeline(allowed_topics=["RAG", "Evaluation", "Guardrails"])
    resp, lat = pipeline.run("Tell me about RAGAS", dummy_rag)
    print(f"Response: {resp} (Latency: {lat:.2f}ms)")
