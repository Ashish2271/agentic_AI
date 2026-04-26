import subprocess
import json

# Import available models
from models import AVAILABLE_MODELS

def call_model(model_name, text, task):
    command = f"python agent.py --model {model_name} --text '{text}' --task '{task}'"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    response = result.stdout.strip()
    return json.loads(response) if response else None

def rate_responses(responses):
    # Placeholder for rating logic
    rated_responses = {key: len(response) for key, response in responses.items()}  # Example rating by response length
    best_model = max(rated_responses, key=rated_responses.get)
    return best_model, responses[best_model]

def ai_council(text, task):
    responses = {}
    for model_name in AVAILABLE_MODELS.keys():
        response = call_model(model_name, text, task)
        if response:
            responses[model_name] = response

    best_model, best_response = rate_responses(responses)
    return best_model, best_response

# Example usage:
# best_model, best_response = ai_council("What is AI?", "Explain the concept of AI.")
# print(f"Best response from {best_model}: {best_response}")
