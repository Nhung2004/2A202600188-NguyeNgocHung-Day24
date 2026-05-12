class OutputGuard:
    def __init__(self):
        pass

    def check_safety(self, response):
        """Simulated Llama Guard 3 check."""
        unsafe_keywords = ["hack", "illegal", "exploit", "password"]
        for word in unsafe_keywords:
            if word in response.lower():
                return False, "Unsafe content detected (Simulated Llama Guard 3)"
        return True, "Safe"

    def validate(self, response):
        is_safe, reason = self.check_safety(response)
        return is_safe, reason
