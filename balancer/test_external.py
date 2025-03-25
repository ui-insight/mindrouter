#
# Run through the list of external models specified in balancer.json
# and test to make sure they respond with valid completions.
#

import json
import requests
import concurrent.futures

# Endpoint and worker configuration
ENDPOINT = "https://mindrouter-api.nkn.uidaho.edu/v1/chat/completions"
NUM_WORKERS = 10
CONFIG_FILE = "balancer.json"

def make_request(model):
    """
    Sends a POST request for the given model.
    Returns a tuple: (model, success_flag, response_data_or_error)
    """
    # Construct a sample payload; modify messages as needed.
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    }
    
    try:
        # Send the request with a timeout (in seconds)
        response = requests.post(ENDPOINT, json=payload, timeout=10)
        
        # Check if the response status code indicates success
        if response.status_code != 200:
            return model, False, f"Unexpected status code: {response.status_code}"
        
        # Attempt to parse the response JSON
        data = response.json()
        
        # Check for an expected field (e.g. "choices")
        if "choices" not in data:
            return model, False, "Missing 'choices' in response"
        
        return model, True, data
    
    except Exception as e:
        # If any exception occurs, consider the request a failure.
        return model, False, f"Exception: {str(e)}"

def main():
    # Load configuration from the config.json file
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
    except Exception as e:
        print(f"Failed to read config.json: {e}")
        return

    # Extract allowed models from the config
    try:
        allowed_models = config["external"][0]["provider"]["allowed_models"]
        model_names = [entry["model"] for entry in allowed_models]
    except (KeyError, IndexError) as e:
        print(f"Error parsing config.json: {e}")
        return

    # List to keep track of models that failed to produce a valid response
    failed_models = []

    # Use a ThreadPoolExecutor to run requests in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all the requests concurrently
        future_to_model = {executor.submit(make_request, model): model for model in model_names}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_model):
            model = future_to_model[future]
            try:
                model, success, result = future.result()
                if success:
                    print(f"Model '{model}' succeeded.")
                else:
                    print(f"Model '{model}' failed: {result}")
                    failed_models.append((model, result))
            except Exception as exc:
                print(f"Model '{model}' generated an exception: {exc}")
                failed_models.append((model, str(exc)))

    # Final report: list all models that did not return a valid response
    print("\nFinal Report: Failed Models")
    if failed_models:
        for model, reason in failed_models:
            print(f"- {model}: {reason}")
    else:
        print("All models responded successfully.")

if __name__ == "__main__":
    main()
