import re
import time
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from langchain_openai import ChatOpenAI

VN_PII = {
    "cccd": r"\b\d{12}\b",
    "phone_vn": r"(\+84|0)\d{9,10}",
    "tax_code": r"\b\d{10}(-\d{3})?\b",
}

class InputGuard:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def scrub_pii(self, text):
        # VN specific
        for name, pattern in VN_PII.items():
            text = re.sub(pattern, f"[{name.upper()}]", text)
        
        # Presidio NER
        results = self.analyzer.analyze(text=text, language="en")
        anonymized_result = self.anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized_result.text

    def validate_topic(self, text, allowed_topics):
        prompt = f"Is this question about one of these topics: {allowed_topics}?\nQuestion: {text}\nAnswer YES or NO only."
        response = self.llm.invoke(prompt).content.strip()
        return response.upper().startswith("YES")

    def sanitize(self, text, allowed_topics):
        start = time.perf_counter()
        
        # 1. PII Redaction
        scrubbed = self.scrub_pii(text)
        
        # 2. Topic Validation
        is_on_topic = self.validate_topic(scrubbed, allowed_topics)
        
        latency_ms = (time.perf_counter() - start) * 1000
        return scrubbed, is_on_topic, latency_ms
