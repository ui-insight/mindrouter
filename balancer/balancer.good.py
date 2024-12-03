from openai import OpenAI
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import requests
import os
import json
import itertools
import random
import logging
from datetime import datetime
import threading
import time
import tiktoken

app = Flask(__name__)
CORS(app)

BALANCE_FACTOR_RANDOM      = 0.25    # Let's add some stochasticity to the selection process!
BALANCE_FACTOR_PRIORITY    = 0.45    # Endpoint speed and priority should play a big role
BALANCE_FACTOR_RUNNING     = 0.05    # Whether the model is loaded already or not should play a role
BALANCE_FACTOR_UTILIZATION = 0.15    # How important is current GPU utilization?
BALANCE_FACTOR_MEMORY      = 0.1     # How important is amount of available memory relative to model size?

REFRESH_INTERVAL = 60     # seconds
NODE_UTILIZATION_THRESHOLD = 85   

CONFIG_FILE = "balancer.json"
STATE_FILE  = "nodestate.json"
LOG_FILE    = "requests.log"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

global_cluster_state_lock = threading.Lock()
global_models_list_lock   = threading.Lock()

with global_cluster_state_lock:
    global_cluster_state = {}

with global_models_list_lock:
    global_models_list = []

global_thread_started = False



# Load configuration
with open(CONFIG_FILE, 'r') as config_file:
    config = json.load(config_file)


# Convert memory string to MB.
def parse_memory(memory_str):
    if "GB" in memory_str:
        return int(float(memory_str.replace(" GB", "")) * 1024)
    elif "MB" in memory_str:
        return int(memory_str.replace(" MB", ""))
    return 0


# Function to tokenize text using tiktoken
def tokenize(text, model_name="gpt-4"):
    encoding = tiktoken.encoding_for_model(model_name)
    return encoding.encode(text)

# Function to fetch data from a given node
def fetch_data_from_node(node_url, endpoint, payload=None, method='GET', stream=False):
    print(f"Fetching data from {node_url}/{endpoint}")  # Debug print
    try:
        if method == 'POST':
            response = requests.post(f"{node_url}/{endpoint}", json=payload, stream=stream)
        else:
            response = requests.get(f"{node_url}/{endpoint}", stream=stream)
        
        if response.status_code == 200:
            if stream:
                return response.iter_lines()
            return response.json()
        else:
            print(f"Response status code: {response.status_code}")  # Debug print
            print(f"Error response: {response.text}")  # Debug print
            return None
    except Exception as e:
        print(f"Failed to fetch data from {node_url}: {e}")
        return None

# Function to check the health of a node
def check_node_health(node_url):
    #print(f"Checking health of {node_url}")  # Debug print
    try:
        response = requests.get(f"{node_url}/health")
        return response.status_code == 200
    except Exception as e:
        print(f"Failed to check health of {node_url}: {e}")
        return False

# Function to get Ollama instance endpoints from a node and associate with agent endpoint
def get_ollama_endpoints(node_url):
    #print(f"Getting Ollama endpoints for {node_url}")  # Debug print
    ollama_info = fetch_data_from_node(node_url, "ollama-info")
    if ollama_info and "ollama_services" in ollama_info:
        endpoints = {service["url"]: node_url for service in ollama_info["ollama_services"]}
        print(f"Found endpoints: {endpoints}")  # Debug print
        return endpoints
    print(f"No Ollama services found for {node_url}")  # Debug print
    return {}



def get_optimal_ollama_instance_with_model(model_name):
    """
    Selects the optimal Ollama instance (endpoint) for the given model based on various criteria
    to balance the load across nodes while reducing latency.

    Args:
        model_name (str): The name of the model to deploy.

    Returns:
        str or None: The URL of the selected endpoint, or None if no viable endpoint is found.
    """
    global global_cluster_state
    print(f"Selecting optimal Ollama instance for model: {model_name}")

    viable_endpoints = []

    # Step 1: Gather all endpoints that have the model available
    for node in global_cluster_state:
        for ollama_info in node.get("ollama_info", []):
            for service in ollama_info.get("ollama_services", []):
                # Check if the model is available on this service
                models_avail = [model['name'] for model in service.get("models_avail", [])]
                if model_name in models_avail:
                    # Initialize variables
                    total_memory = 0
                    used_memory = 0
                    gpu_utilizations = []

                    # Collect GPU info for GPUs used by this service
                    gpu_indices = service.get("gpu_indices", [])
                    for gpu in node.get("gpu_info", [])[0].get("gpus", []):
                        if gpu['index'] in gpu_indices:
                            total_memory += parse_memory(gpu['memory_total'])
                            used_memory += parse_memory(gpu['memory_used'])
                            gpu_utilizations.append(gpu['utilization'])

                    available_memory = total_memory - used_memory
                    priority = service.get("priority", 10)  # Default to 10 if not specified
                    url = service["url"]

                    # Get the memory requirement of the model
                    model_info = next((m for m in service.get("models_avail", []) if m['name'] == model_name), None)
                    if model_info:
                        model_memory_required = parse_memory(model_info['size'])
                    else:
                        model_memory_required = 0  # Should not happen since we checked model_name in models_avail

                    # Check if the model is currently running
                    running_models = [running_model['name'] for running_model in service.get("models_running", [])]
                    is_running = model_name in running_models

                    # Calculate average GPU utilization
                    average_gpu_utilization = sum(gpu_utilizations) / len(gpu_utilizations) if gpu_utilizations else 0

                    # Create endpoint_info
                    endpoint_info = {
                        "service_name": service["service_name"],
                        "url": url,
                        "total_gpu_memory_MB": total_memory,
                        "available_gpu_memory_MB": available_memory,
                        "model_memory_required_MB": model_memory_required,
                        "average_gpu_utilization": average_gpu_utilization,
                        "running": is_running,
                        "priority": priority
                    }

                    # Only consider endpoints where total GPU memory is sufficient
                    if total_memory >= model_memory_required:
                        viable_endpoints.append(endpoint_info)
                        print(f"Found viable endpoint: {endpoint_info}")
                    else:
                        print(f"Endpoint {url} skipped: total GPU memory ({total_memory}MB) less than model requirement ({model_memory_required}MB)")

    # Step 2: Check if any viable endpoints are available
    if not viable_endpoints:
        print("No viable endpoints found with sufficient total GPU memory to run the model.")
        return None

    # Step 3: Consider all viable endpoints together
    print("Scoring all viable endpoints based on various factors.")

    scored_endpoints = []
    for ep in viable_endpoints:
        # Normalize the variables
        normalized_priority = (10 - ep["priority"]) / 10  # Normalized to [0,1]; higher is better
        available_memory_ratio = ep["available_gpu_memory_MB"] / ep["model_memory_required_MB"]
        normalized_available_memory = min(available_memory_ratio, 1)  # Capped at 1; higher is better
        normalized_utilization = (100 - ep["average_gpu_utilization"]) / 100  # Normalized to [0,1]; higher is better
        normalized_running = 1 if ep["running"] else 0  # 1 if running, 0 if not
        random_factor = random.random()  # Random number between [0,1]


        # Calculate the score with equal weights
        score = (
            ( random_factor               * BALANCE_FACTOR_RANDOM      ) +
            ( normalized_priority         * BALANCE_FACTOR_PRIORITY    ) +
            ( normalized_running          * BALANCE_FACTOR_RUNNING     ) +
            ( normalized_utilization      * BALANCE_FACTOR_UTILIZATION ) +
            ( normalized_available_memory * BALANCE_FACTOR_MEMORY      )
        )

        # SHENEMAN - need to address this better
        #
        # Exclude endpoints that are too busy
        #if ep["average_gpu_utilization"] >= NODE_UTILIZATION_THRESHOLD:
        #    print(f"Endpoint {ep['url']} is too busy (GPU utilization {ep['average_gpu_utilization']}%). Skipping.")
        #    continue

        scored_endpoints.append((score, ep))

        # Debugging output
        print(
            f"Endpoint {ep['url']} - "
            f"Normalized Priority: {normalized_priority:.2f}, "
            f"Normalized Available Memory: {normalized_available_memory:.2f}, "
            f"Normalized Utilization: {normalized_utilization:.2f}, "
            f"Running: {normalized_running}, "
            f"Random Factor: {random_factor:.2f}, "
            f"Score: {score:.2f}"
        )

    # Step 4: Check if any endpoints were scored
    if not scored_endpoints:
        print("No viable endpoints found after scoring (all may be too busy).")
        return None

    # Step 5: Select the endpoint with the highest score
    best_endpoint = max(scored_endpoints, key=lambda x: x[0])[1]
    print(f"Selected endpoint: {best_endpoint['url']} with score: {max(scored_endpoints)[0]:.2f}")
    return best_endpoint["url"]




def get_running_models(instance_url):
    running_models = fetch_data_from_node(instance_url, "api/ps")
    if running_models and "models" in running_models:
        return [model["name"] for model in running_models["models"]]
    return []

# Function to get GPU utilization of an instance
def get_gpu_utilization(agent_url):
    gpu_info = fetch_data_from_node(agent_url, "gpu-info")
    if gpu_info:
        try:
            print(f"GPU info received: {gpu_info}")  # Debug print
            
            gpus = gpu_info.get("gpus", [])
            if not gpus:
                print("No GPU information available.")
                return float('inf')

            # Calculate average utilization across all GPUs
            total_utilization = sum(gpu.get("utilization", 0) for gpu in gpus)
            average_utilization = total_utilization / len(gpus)

            return average_utilization
        except KeyError as e:
            print(f"Error parsing GPU utilization: {e}")
            return float('inf')
    else:
        print(f"Failed to retrieve GPU utilization from {agent_url}.")
        return float('inf')  # Return a high value if GPU utilization cannot be fetched


# Function to log requests in jsonl format
def log_request(log_data):
    with open(LOG_FILE, 'a') as log_file:
        log_file.write(json.dumps(log_data) + '\n')

# Function to extract prompt length from the request data
def extract_prompt_length(messages):
    prompt = ' '.join([msg.get('content', '') for msg in messages])
    return len(prompt.split())


# function to determine if the user-specified model is from an external provider
def is_model_external(config, model_name):
    if "external" in config:
        for provider in config["external"]:
            if "provider" in provider and "allowed_models" in provider["provider"]:
                for model_info in provider["provider"]["allowed_models"]:
                    if model_info.get("model") == model_name:
                        return True
    return False


# returns the provider name and base url of the given model
def find_provider_by_model(config, model_name):
    if "external" in config:
        for provider in config["external"]:
            if "provider" in provider and "allowed_models" in provider["provider"]:
                for model_info in provider["provider"]["allowed_models"]:
                    if model_info.get("model") == model_name:
                        return provider["provider"].get("name"), provider["provider"].get("base_url")
    return None, None



def convert_ollama_to_openai_request(data):
    """Convert Ollama format request to OpenAI format"""
    
    # Extract the prompt from Ollama format
    prompt = data.get('prompt', '')
    system_prompt = data.get('system', '')
    
    # Build OpenAI-style messages array
    messages = []
    
    # Add system message if present
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    # Add user message
    messages.append({
        "role": "user",
        "content": prompt
    })
    
    # Build OpenAI-compatible request
    openai_request = {
        "model": data.get('model'),
        "messages": messages,
        "stream": data.get('stream', False),
        "temperature": data.get('temperature', 0.7),
        "max_tokens": data.get('context_length', 32000)
    }
    
    return openai_request

def convert_openai_to_ollama_response(openai_response, stream=False):
    """Convert OpenAI format response to Ollama format"""
    
    if stream:
        # Handle streaming response
        content = openai_response['choices'][0]['delta'].get('content', '')
        return {
            "model": openai_response.get('model', ''),
            "created_at": openai_response.get('created', ''),
            "response": content,
            "done": openai_response['choices'][0].get('finish_reason') == "stop"
        }
    else:
        # Handle normal response
        content = openai_response['choices'][0]['message']['content']
        return {
            "model": openai_response.get('model', ''),
            "created_at": openai_response.get('created', ''),
            "response": content,
            "done": True,
            "total_duration": 0,  # OpenAI doesn't provide this
            "load_duration": 0,   # OpenAI doesn't provide this
            "prompt_eval_count": openai_response['usage'].get('prompt_tokens', 0),
            "eval_count": openai_response['usage'].get('completion_tokens', 0),
            "context_length": openai_response['usage'].get('total_tokens', 0)
        }



def process_single_request(request_data):
    print("********** BEGIN REQUEST ***********")

    global global_thread_started
    global global_cluster_state

    if not global_thread_started:
        with global_cluster_state_lock:
            global_cluster_state = aggregate_hierarchical_data()

    data = request_data['data']
    print("DATA: ", data, flush=True)
    model_name = request_data['model_name']
    client_ip = request_data['client_ip']
    request_time = request_data['request_time']
    request_path = request_data['request_path']

    print("In process_single_request()")
    print("CONFIG External: ", config["external"])

    print(f"Processing request for model: {model_name}")

    external_model_flag = is_model_external(config, model_name)

    if external_model_flag:
        print("MODEL IS EXTERNAL!") 
        model_provider, model_base_url = find_provider_by_model(config, model_name)
        print("  MODEL Provider: ", model_provider)
        print("  MODEL Base URL: ", model_base_url)
        print("\n")
       
        instance_url = model_base_url
    else:
        instance_url = get_optimal_ollama_instance_with_model(model_name)

    if not instance_url:
        print(f"No instances available for model {model_name}")
        yield '{"error": "No instances available for model"}\n'
        return

    print("INSTANCE URL: ", instance_url)

    start_time = time.time()

    # Determine the backend endpoint based on the request path
    if '/v1/chat/completions' in request_path:
        # Check if the "stream" attribute is true
        if data.get("stream", False):  # Default to False if "stream" is not present
            format_sse = True
        else:
            format_sse = False
        backend_endpoint = 'v1/chat/completions'
        is_v1_endpoint = True  # Flag to identify /v1/chat/completions
    elif '/api/embeddings' in request_path:
        backend_endpoint = 'api/embeddings'
        format_sse = False  # No SSE formatting needed
        is_v1_endpoint = False
    else:
        backend_endpoint = 'api/chat'
        format_sse = False  # No SSE formatting needed
        is_v1_endpoint = False

    response_content = []
    url = instance_url + "/" + backend_endpoint
    line_cnt = 0  # Initialize line count

    if external_model_flag:
        print("Trying to call external API: ", url, data)
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY
        )

        stream_mode = data.get('stream', False)

        try:
            completion = client.chat.completions.create(
                model=data['model'],
                messages=data['messages'],
                stream=stream_mode,
                temperature=data.get('temperature', 0.2),
                max_tokens=data.get('max_tokens', 32000)
            )

            if stream_mode:
                # Handle streaming response
                for chunk in completion:
                    line_cnt += 1
                    if chunk:
                        chunk_dict = {
                            "id": chunk.id,
                            "object": chunk.object,
                            "created": chunk.created,
                            "model": chunk.model,
                            "choices": [{
                                "index": choice.index,
                                "delta": {
                                    "role": choice.delta.role if choice.delta.role else None,
                                    "content": choice.delta.content if choice.delta.content else None
                                },
                                "finish_reason": choice.finish_reason
                            } for choice in chunk.choices]
                        }
                        response_content.append(chunk_dict)
                    
                        if format_sse:
                            yield f"data: {json.dumps(chunk_dict)}\n\n"
                        else:
                            yield f"{json.dumps(chunk_dict)}\n"
            
                # Send the [DONE] message at the end if using SSE format
                if format_sse:
                    yield "data: [DONE]\n\n"
            else:
                # Handle non-streaming response
                completion_dict = {
                    "id": completion.id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": completion.model,
                    "choices": [{
                        "index": idx,
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content
                        },
                        "finish_reason": choice.finish_reason
                    } for idx, choice in enumerate(completion.choices)],
                    "usage": {
                        "prompt_tokens": completion.usage.prompt_tokens,
                        "completion_tokens": completion.usage.completion_tokens,
                        "total_tokens": completion.usage.total_tokens
                    } if completion.usage else {}
                }
                
                response_content.append(completion_dict)
                line_cnt = 1
                
                if format_sse:
                    yield f"data: {json.dumps(completion_dict)}\n\n"
                    yield "data: [DONE]\n\n"
                else:
                    yield f"{json.dumps(completion_dict)}\n"

        except Exception as e:
            print(f"Error during external API call: {str(e)}")
            if format_sse:
                yield f"data: [ERROR] API call failed: {str(e)}\n\n"
                yield 'data: [DONE]\n\n'
            else:
                yield f'{{"error": "API call failed: {str(e)}"}}\n'
            
    else:
        # Handle internal API calls
        try:
            response_stream = fetch_data_from_node(instance_url, backend_endpoint, payload=data, method='POST', stream=True)
            
            if response_stream:
                for line in response_stream:
                    if line:
                        line = line.decode('utf-8').strip()

                        if format_sse:
                            # For SSE format (used by /v1/chat/completions)
                            if line.startswith("data:"):
                                line = line[len("data:"):].strip()  # Remove "data:" prefix
                                line_cnt += 1

                            if line == "[DONE]":
                                print("Received end-of-stream marker")
                                yield 'data: [DONE]\n\n'
                                break

                            try:
                                parsed_line = json.loads(line)
                                response_content.append(parsed_line)
                                yield f"data: {json.dumps(parsed_line)}\n\n"
                            except json.JSONDecodeError:
                                print(f"Non-JSON or empty line received: {line}")
                                yield f"data: {line}\n\n"
                        else:
                            # For /api/chat and /api/embed (no SSE format)
                            try:
                                parsed_line = json.loads(line)
                                response_content.append(parsed_line)
                                yield f"{json.dumps(parsed_line)}\n"
                            except json.JSONDecodeError:
                                print(f"Non-JSON or empty line received: {line}")
                                yield f"{line}\n"

            else:
                print("No response stream available from node")
                if format_sse:
                    yield 'data: {"error": "Failed to get response from LLM instance"}\n\n'
                else:
                    yield '{"error": "Failed to get response from LLM instance"}\n'

        except Exception as e:
            print(f"Error during streaming: {str(e)}")
            if format_sse:
                yield f"data: [ERROR] Streaming interrupted: {str(e)}\n\n"
                yield 'data: [DONE]\n\n'
            else:
                yield f'{{"error": "Streaming interrupted: {str(e)}"}}\n'

    end_time = time.time()

    # Initialize token counts
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    # Extract token counts from the final response
    if response_content:
        final_response = response_content[-1]

        if is_v1_endpoint:
            # For /v1/chat/completions endpoint
            if "usage" in final_response:
                prompt_tokens = final_response["usage"].get("prompt_tokens", 0)
                completion_tokens = final_response["usage"].get("completion_tokens", 0)
                total_tokens = final_response["usage"].get("total_tokens", prompt_tokens + completion_tokens)
            else:
                messages = data.get('messages', [])
                print(str(messages))
                prompt_tokens = len(tokenize(str(messages))) 
                completion_tokens = line_cnt
                total_tokens = prompt_tokens + completion_tokens
        else:
            # For /api/chat endpoint
            prompt_tokens = final_response.get("prompt_eval_count", 0)
            completion_tokens = final_response.get("eval_count", 0)
            total_tokens = prompt_tokens + completion_tokens

    response_time_ms = int((end_time - start_time) * 1000)

    log_data = {
        "timestamp": request_time,
        "ip_address": client_ip,
        "model": model_name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "inference_server": instance_url,
        "model_loaded": True,
        "response_time_ms": response_time_ms
    }
    log_request(log_data)

    print(f"***END REQUEST: {response_time_ms} msec")


# Function to aggregate data from all nodes
def aggregate_data():
    print("Aggregating data from all nodes")  # Debug print
    gpu_info_list = []
    ollama_info_list = []

    for node in config["nodes"]:
        print(f"Fetching data from node: {node['url']}")  # Debug print
        node_gpu_info = fetch_data_from_node(node["url"], "gpu-info")
        node_ollama_info = fetch_data_from_node(node["url"], "ollama-info")
        
        if node_gpu_info:
            gpu_info_list.append({"node": node["url"], "gpu_info": node_gpu_info})
        if node_ollama_info:
            ollama_info_list.append({"node": node["url"], "ollama_info": node_ollama_info})

    print(f"Aggregated data: {len(gpu_info_list)} GPU info, {len(ollama_info_list)} Ollama info")  # Debug print
    return gpu_info_list, ollama_info_list

# Function to collect and aggregate hierarchical information
def aggregate_hierarchical_data():
    #print("Aggregating hierarchical data")  # Debug print
    aggregated_info = []
    for node in config["nodes"]:
        node_url = node["url"]
        #print(f"Collecting data for node: {node_url}")  # Debug print
        node_health = check_node_health(node_url)
        node_info = {
            "node": node_url,
            "health": node_health,
            "ollama_info": [],
            "gpu_info": []
        }
        ollama_info = fetch_data_from_node(node_url, "ollama-info")
        if ollama_info:
            node_info["ollama_info"].append(ollama_info)
        gpu_info = fetch_data_from_node(node_url, "gpu-info")
        if gpu_info:
            node_info["gpu_info"].append(gpu_info)
        aggregated_info.append(node_info)
    print(f"Aggregated info for {len(aggregated_info)} nodes")  # Debug print
    return aggregated_info

# Function to get models from all Ollama instances
def get_models_from_ollama_instances():

    #print("Getting models from all Ollama instances")  # Debug print
    all_models = []
    models_set = set()

    for node in config["nodes"]:
        ollama_endpoints = get_ollama_endpoints(node["url"])
        for endpoint in ollama_endpoints:
            #print(f"Fetching models from endpoint: {endpoint}")  # Debug print
            node_models = fetch_data_from_node(endpoint, "api/tags")
            if node_models:
                for model in node_models.get("models", []):
                    if model["name"] not in models_set:
                        models_set.add(model["name"])
                        all_models.append(model)

    # Sort the models by name in alphabetical order
    all_models_sorted = sorted(all_models, key=lambda x: x["name"])
    print(f"Found {len(all_models_sorted)} models across {len(config['nodes'])} endpoints.")

    return all_models_sorted


@app.route('/', methods=['GET', 'HEAD'])
def index():
    return Response('Ollama is running', status=200, content_type='text/plain')



def keep_alive_generator(main_generator):
    keep_alive_interval = 15  # Send keep-alive every 15 seconds
    last_keep_alive = time.time()

    for chunk in main_generator:
        yield chunk
        current_time = time.time()
        if current_time - last_keep_alive > keep_alive_interval:
            yield ' '  # Send a space as a keep-alive
            last_keep_alive = current_time

@app.route('/api/chat', methods=['POST'])
def chat_completion():
    print("/API/CHAT")
    try:
        data = request.json
        data.pop('keep_alive', None)
        model_name = data.get('model')
        client_ip = request.remote_addr
        request_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        request_path = request.path

        def generate():
            try:
                for chunk in keep_alive_generator(process_single_request({
                    "data": data,
                    "model_name": model_name,
                    "client_ip": client_ip,
                    "request_time": request_time,
                    "request_path": request_path
                })):
                    yield chunk
            except GeneratorExit:
                print("Client closed connection prematurely")
            except Exception as e:
                print(f"Unexpected error in streaming: {str(e)}")
                yield f'{{"error": "Unexpected error: {str(e)}"}}\n'

        return Response(generate(), mimetype='application/json', content_type='application/json')
    except Exception as e:
        print(f"Error in chat_completion: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate():
    print("/API/GENERATE")
    try:
        data = request.json
        model_name = data.get('model')
        client_ip = request.remote_addr
        request_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def generate():
            for chunk in keep_alive_generator(process_single_request({
                "data": data,
                "model_name": model_name,
                "client_ip": client_ip,
                "request_time": request_time
            })):
                yield chunk

        return Response(generate(), mimetype='application/json', content_type='application/json')
    except Exception as e:
        print(f"Error in generate: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/v1/chat/completions', methods=['POST'])
@app.route('/chat/completions', methods=['POST'])
def openai_chat_completions():
    try:
        print("Received OpenAI-style chat completion request")  # Debug print
        data = request.json
        model_name = data.get('model')
        client_ip = request.remote_addr
        request_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        request_path = request.path  # Capture the request path
        print("request_path = ", request_path)

        if not request.path.startswith('/v1'):
            request_path = f'/v1{request.path}'
        
        # Check if 'stream' is True in the request JSON
        if data.get('stream') is True:
            print("Stream Is True")  # Print if stream is True

            # Assuming `data` contains 'messages' which is a list of chat messages
            system_prompt = ""
            user_prompt = ""
            
            for message in data.get('messages', []):
                if message.get('role') == 'system':
                    system_prompt = message.get('content', "")
                elif message.get('role') == 'user':
                    user_prompt = message.get('content', "")
            
            # Tokenize the system and user prompts
            system_tokens = tokenize(system_prompt, "gpt-4")
            user_tokens   = tokenize(user_prompt, "gpt-4")

            # Print the number of tokens in each prompt
            print(f"System Prompt Tokens: {len(system_tokens)}")
            print(f"User Prompt Tokens: {len(user_tokens)}")
        
        def generate():
            for chunk in process_single_request({
                "data": data,
                "model_name": model_name,
                "client_ip": client_ip,
                "request_time": request_time,
                "request_path": request_path  # Include the request path in the request data
            }):
                yield chunk

        return Response(generate(), mimetype='application/json', content_type='application/json')
    except Exception as e:
        print(f"Error in openai_chat_completions: {str(e)}")  # Debug print
        return jsonify({"error": str(e)}), 500

@app.route('/api/embeddings', methods=['POST'])
def embed_text():
    try:
        print("Received embedding request")
        data = request.json
        model_name = data.get('model')
        client_ip = request.remote_addr
        request_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        request_path = request.path

        def generate():
            try:
                for chunk in process_single_request({
                    "data": data,
                    "model_name": model_name,
                    "client_ip": client_ip,
                    "request_time": request_time,
                    "request_path": request_path
                }):
                    yield chunk
            except GeneratorExit:
                print("Client closed connection prematurely")
            except Exception as e:
                print(f"Unexpected error in streaming: {str(e)}")
                yield f'{{"error": "Unexpected error: {str(e)}"}}\n'

        return Response(generate(), mimetype='application/json', content_type='application/json')
    except Exception as e:
        print(f"Error in embed_text(): {str(e)}")
        return jsonify({"error": str(e)}), 500


# Endpoint to get GPU information from all nodes
@app.route('/gpu-info', methods=['GET'])
def gpu_info():
    try:
        print("Received request for GPU info")  # Debug print
        gpu_info_list, _ = aggregate_data()
        print(f"Returning GPU info for {len(gpu_info_list)} nodes")  # Debug print
        pretty_json = json.dumps(gpu_info_list, indent=4)
        return Response(pretty_json, mimetype='application/json')
    except Exception as e:
        print(f"Error in gpu_info: {str(e)}")  # Debug print
        error_json = json.dumps({"error": str(e)}, indent=4)
        return Response(error_json, mimetype='application/json'), 500

# Endpoint to get Ollama information from all nodes
@app.route('/ollama-info', methods=['GET'])
def ollama_info():
    try:
        print("Received request for Ollama info")  # Debug print
        _, ollama_info_list = aggregate_data()
        print(f"Returning Ollama info for {len(ollama_info_list)} nodes")  # Debug print
        pretty_json = json.dumps(ollama_info_list, indent=4)
        return Response(pretty_json, mimetype='application/json')
    except Exception as e:
        print(f"Error in ollama_info: {str(e)}")  # Debug print
        error_json = json.dumps({"error": str(e)}, indent=4)
        return Response(error_json, mimetype='application/json'), 500

# Endpoint to list all nodes with their health status
@app.route('/list-nodes', methods=['GET'])
def list_nodes():
    try:
        print("Received request to list nodes")  # Debug print
        nodes_status = [{"node": node["url"], "health": check_node_health(node["url"])} for node in config["nodes"]]
        print(f"Returning status for {len(nodes_status)} nodes")  # Debug print
        pretty_json = json.dumps({"nodes": nodes_status}, indent=4)
        return Response(pretty_json, mimetype='application/json')
    except Exception as e:
        print(f"Error in list_nodes: {str(e)}")  # Debug print
        error_json = json.dumps({"error": str(e)}, indent=4)
        return Response(error_json, mimetype='application/json'), 500

# Endpoint to collect and aggregate hierarchical information
@app.route('/collect-info', methods=['GET'])
def collect_info():
    try:
        print("Received request to collect hierarchical info")  # Debug print
        aggregated_info = aggregate_hierarchical_data()
        pretty_json = json.dumps(aggregated_info, indent=4)
        return Response(pretty_json, mimetype='application/json')
    except Exception as e:
        print(f"Error in collect_info: {str(e)}")  # Debug print
        error_json = json.dumps({"error": str(e)}, indent=4)
        return Response(error_json, mimetype='application/json'), 500


# Endpoint to get models from all Ollama instances and de-duplicate
@app.route('/api/tags', methods=['GET'])
def get_models():
    global global_models_list, config

    try:
        print("Received request for model tags")  # Debug print

        with global_models_list_lock:
            # Copy the global models list
            all_models = global_models_list.copy()

        # Add external models
        if "external" in config:
            for external_provider in config["external"]:
                if "provider" in external_provider and "allowed_models" in external_provider["provider"]:
                    for model_info in external_provider["provider"]["allowed_models"]:
                        # Create an entry for the external model with just "name" and "model"
                        external_model_entry = {
                            "name": model_info["model"],
                            "model": model_info["model"]
                        }
                        all_models.append(external_model_entry)  # Add to the aggregated list

        # Create the response
        pretty_json = json.dumps({"models": all_models}, indent=4)
        return Response(pretty_json, mimetype='application/json')
    except Exception as e:
        print(f"Error in get_models: {str(e)}")  # Debug print
        error_json = json.dumps({"error": str(e)}, indent=4)
        return Response(error_json, mimetype='application/json'), 500


def save_state(cluster_state, models_list, state_file):
    combined_state = {
        "cluster_state": cluster_state,
        "models_list": models_list
    }
    
    try:
        with open("tmpstate.json", 'w') as f:
            json.dump(combined_state, f, indent=2)
        os.rename("tmpstate.json", state_file) 
    except Exception as e:
        print(f"Error saving state: {str(e)}")


def load_state(state_file):
    cluster_state = {}
    models_list = {}
    
    try:
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                combined_state = json.load(f)
                cluster_state = combined_state.get("cluster_state", {})
                models_list = combined_state.get("models_list", {})
    except Exception as e:
        print(f"Error loading state: {str(e)}")
        
    return cluster_state, models_list



def start_background_thread():

    global global_thread_started
    global global_cluster_state
    global global_models_list

    print("start_background_thread()")

    if os.path.exists(STATE_FILE):
        print(f"Found existing state file at {STATE_FILE}, loading...")
        global_cluster_state, global_models_list = load_state(STATE_FILE)
    else:
        print(f"No state file found at {STATE_FILE}, starting with empty state")

    if not global_thread_started:
        global_thread_started = True
        threading.Thread(target=update_ollama_state, args=(REFRESH_INTERVAL,), daemon=True).start()


def update_ollama_state(delay=REFRESH_INTERVAL):
    global global_cluster_state
    global global_models_list

    while True:
        with global_cluster_state_lock:
            global_cluster_state = aggregate_hierarchical_data()

        with global_models_list_lock:
            global_models_list = get_models_from_ollama_instances()

        save_state(global_cluster_state, global_models_list, STATE_FILE)

        time.sleep(delay)  # Wait for the specified delay before updating again


start_background_thread()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8009, threaded=True)

