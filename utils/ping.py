import requests
import time

# List of service URLs
SERVICE_URLS = [
    'http://calvin.nkn.uidaho.edu:8001/v1/',
    'http://aurora.nkn.uidaho.edu:8001/v1/',
    'http://wintermute.nkn.uidaho.edu:8001/v1/',
    'http://webbyg2.nkn.uidaho.edu:8001/v1/',
]

# Function to call the LLM with a simple prompt and log the response status
def check_llm_response(service_url, model_name, prompt):
    try:
        response = requests.post(
            f"{service_url}chat/completions",
            json={
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 100
            },
            timeout=120  # Set a timeout for the request
        )

        if response.status_code == 200:
            data = response.json()
            if data and data.get("choices"):
                response_text = data["choices"][0]["message"]["content"]
                print(f"Service: {service_url}, Model: {model_name}, Response: {response_text}")
            else:
                print(f"Service: {service_url}, Model: {model_name}, No valid response from the API.")
        elif response.status_code == 404:
            print(f"Service: {service_url}, Model: {model_name}, Error: Model not found (404).")
        else:
            print(f"Service: {service_url}, Model: {model_name}, HTTP Error: {response.status_code}")

    except requests.exceptions.ConnectionError as e:
        print(f"Service: {service_url}, Model: {model_name}, Connection Error: {str(e)}")
    except requests.exceptions.Timeout as e:
        print(f"Service: {service_url}, Model: {model_name}, Timeout Error: {str(e)}")
    except requests.exceptions.RequestException as e:
        print(f"Service: {service_url}, Model: {model_name}, Request Error: {str(e)}")

# Main function
def main():
    model_name = "llama3.1:70b"  # Replace with your model name
    prompt = "What is the capital of France?"  # Example prompt

    for service_url in SERVICE_URLS:
        print(f"Testing service: {service_url}")
        check_llm_response(service_url, model_name, prompt)
        print("-" * 50)  # Separator for clarity in the output

# Run the check
if __name__ == "__main__":
    main()

