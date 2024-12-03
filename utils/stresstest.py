import requests
import time
import random
import openai
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up the Ollama API base URL and API key
OLLAMA_BASE_URL = 'https://mindrouter-api.nkn.uidaho.edu/v1/'
API_KEY = 'ollama'  # Required but ignored for Ollama's API

# Initialize the OpenAI client for Ollama
client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=API_KEY)

# Function to get the list of models available
def get_available_models():
    response = requests.get('https://mindrouter-api.nkn.uidaho.edu/api/tags')
    if response.status_code == 200:
        models = response.json().get("models", [])
        # Extract names and filter out models that start with "EMBED"
        return [model["name"] for model in models if not model["name"].startswith("EMBED")]
    else:
        print("Error: Unable to retrieve models.")
        return []

# Function to call a model with a test prompt, measure response time, and print the response
def call_model_multiple_times(model_name, prompt, repetitions):
    for i in range(repetitions):
        start_time = time.time()
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1000
            )

            end_time = time.time()

            # Check if response is valid
            if response and response.choices:
                response_time = end_time - start_time
                response_text = response.choices[0].message.content
                print(f"Model: {model_name}, Call {i + 1}/{repetitions}, Response Time: {response_time:.2f} seconds\nResponse Len: {len(response_text)}\n")
            else:
                print(f"Model: {model_name}, Call {i + 1}/{repetitions}, Error: No valid response from the API")

        except Exception as e:
            print(f"Model: {model_name}, Call {i + 1}/{repetitions}, Error: {str(e)}")

# Main function to run the stress test
def stress_test_load_balancer():
    models = get_available_models()
    if not models:
        print("No models available for testing.")
        return

    # Sample prompt for testing
    test_prompt = "What is the capital of France?"

    # Set up a thread pool with 10 workers
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for _ in range(10):
            # Randomly select a model
            model = random.choice(models)

            # Submit 50 calls for the selected model in rapid succession
            futures.append(executor.submit(call_model_multiple_times, model, test_prompt, 500))

        # Iterate over the futures as they complete
        for future in as_completed(futures):
            try:
                # Retrieve result (if any) to catch exceptions raised in threads
                future.result()
            except Exception as e:
                print(f"Error during execution: {e}")

# Run the stress test
if __name__ == "__main__":
    stress_test_load_balancer()

