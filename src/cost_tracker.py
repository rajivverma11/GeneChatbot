from typing import Dict

MODEL_COST_PER_1K = {
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-3.5-turbo": {"prompt": 0.001, "completion": 0.002},
}

class CostTracker:
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def update(self, usage: Dict[str, int]):
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)

    def get_cost(self) -> float:
        model_cost = MODEL_COST_PER_1K.get(self.model_name, MODEL_COST_PER_1K["gpt-4o"])
        prompt_cost = (self.prompt_tokens / 1000) * model_cost["prompt"]
        completion_cost = (self.completion_tokens / 1000) * model_cost["completion"]
        return round(prompt_cost + completion_cost, 6)

    def report(self):
        total_cost = self.get_cost()
        print(f"🔢 Prompt Tokens: {self.prompt_tokens}")
        print(f"🧠 Completion Tokens: {self.completion_tokens}")
        print(f"💰 Estimated Total Cost: ${total_cost}")
