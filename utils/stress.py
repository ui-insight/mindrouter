import requests
import concurrent.futures
import time
import argparse
import random

# Define the API endpoints
COMPLETIONS_ENDPOINT = "http://aurora:8009/v1/chat/completions"

# Define the list of models to test
MODELS = [
    "mistral-nemo:12b", "gemma2:27b", "gemma2:9b", "llama3.1:70b", "llama3.1:8b",
    "command-r:35b-v0.1-q5_1", "aya:35b", "aya:8b", "qwen2:0.5b", "qwen2:1.5b",
    "qwen2:7b", "qwen2:72b", "qwen:110b", "command-r-plus:latest", "mixtral:8x22b",
    "mixtral:instruct", "gemma:latest", "mistral:latest", "dolphin-mixtral:8x7b",
    "dolphin-mixtral:8x22b", "phi3:latest", "llama2-uncensored:7b", "samantha-mistral:7b",
    "llama3-gradient:8b", "falcon:7b", "meditron:70b", "falcon:40b", "medllama2:7b"
]


def send_completion_request(model_name):
    """Send an OpenAI-compatible completion request"""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "write a sophisticated chess program in python"}],
        "max_tokens": 10,
        "temperature": 0.7
    }

    response = requests.post(COMPLETIONS_ENDPOINT, json=payload)

    if response.status_code == 200:
        response_data = response.json()
        # Assuming the text is in the 'choices' field
        if 'choices' in response_data and len(response_data['choices']) > 0:
            completion_text = response_data['choices'][0]['message']['content']
            print(model_name, completion_text)
        else:
            print(model_name, "No choices found in the response.")
    else:
        print(model_name, "Request failed with status code:", response.status_code)

    return response.status_code

def stress_test(num_requests, delay):
    """Perform multiple concurrent requests with a specified delay"""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for _ in range(num_requests):
            model_name = random.choice(MODELS)
            futures.append(executor.submit(send_completion_request, model_name))
            time.sleep(delay)
        results = [future.result() for future in futures]
    return results

def main():
    parser = argparse.ArgumentParser(description="LLM Load Balancer Stress Tester")
    parser.add_argument("-n", "--num-requests", type=int, default=10, help="Number of requests to send")
    parser.add_argument("-d", "--delay", type=float, default=0.1, help="Delay between requests in seconds")
    args = parser.parse_args()

    results = stress_test(args.num_requests, args.delay)
    print(f"Results: {results}")

if __name__ == "__main__":
    main()
