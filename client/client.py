from flask import Flask, jsonify, Response
import json
import subprocess
import os
import platform
import psutil
import socket
import pynvml

app = Flask(__name__)

# Initialize NVML
pynvml.nvmlInit()

# Load configuration
with open('client.json', 'r') as config_file:
    config = json.load(config_file)

def get_gpu_info(index):
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_total = memory_info.total // 1024 // 1024  # bytes to MB
        memory_used = memory_info.used // 1024 // 1024  # bytes to MB
        
        return {
            "index": index,
            "uuid": uuid,
            "name": name,
            "temperature": temperature,
            "utilization": utilization,
            "memory_total": f"{memory_total} MB",
            "memory_used": f"{memory_used} MB",
            "processes": get_processes(handle)
        }
    except pynvml.NVMLError as e:
        print(f"Failed to get GPU info for index {index}: {e}")
        return None

def get_processes(handle):
    process_info = []
    try:
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        for process in processes:
            process_info.append({
                "pid": process.pid,
                "process_name": get_process_name(process.pid),
                "used_memory": process.usedGpuMemory // 1024 // 1024
            })
    except pynvml.NVMLError as e:
        print(f"Failed to get running processes: {e}")

    return process_info

def get_process_name(pid):
    try:
        with open(f"/proc/{pid}/comm", 'r') as file:
            return file.readline().strip()
    except Exception:
        return "Unknown"

def get_ollama_version(command, port):
    try:
        env = os.environ.copy()
        env['OLLAMA_HOST'] = f'0.0.0.0:{port}'
        result = subprocess.run([command, '--version'], capture_output=True, text=True, env=env)
        if result.returncode == 0:
            return result.stdout.strip().split()[-1]  # Extract just the version number
        else:
            return "Unknown"
    except Exception as e:
        return f"Error: {e}"

def get_ollama_list(command, port):
    try:
        env = os.environ.copy()
        env['OLLAMA_HOST'] = f'0.0.0.0:{port}'
        result = subprocess.run([command, 'list'], capture_output=True, text=True, env=env)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            models = []
            for line in lines[1:]:  # Skip header
                parts = line.split()
                models.append({
                    "name": parts[0],
                    "id": parts[1],
                    "size": f"{parts[2]} {parts[3]}",  # Combine size and unit
                    "modified": ' '.join(parts[4:])  # Handle spaces in the modified date
                })
            return models
        else:
            return []
    except Exception as e:
        print(f"Failed to list models: {e}")
        return []

def get_ollama_show(command, port, model_name):
    try:
        env = os.environ.copy()
        env['OLLAMA_HOST'] = f'0.0.0.0:{port}'
        result = subprocess.run([command, 'show', model_name], capture_output=True, text=False, env=env)
        if result.returncode == 0:
            # Decode the output with UTF-8 instead of using the default ASCII
            output = result.stdout.decode('utf-8')
            details = {}
            lines = output.strip().split('\n')
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    details[key] = value
            return details
        else:
            return {}
    except Exception as e:
        print(f"Failed to show model {model_name}: {e}")
        return {}

def get_ollama_ps(command, port):
    try:
        env = os.environ.copy()
        env['OLLAMA_HOST'] = f'0.0.0.0:{port}'
        result = subprocess.run([command, 'ps'], capture_output=True, text=True, env=env)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            processes = []
            for line in lines[1:]:  # Skip header
                parts = line.split()
                size = f"{parts[2]} {parts[3]}"  # Combine size and unit
                processor_until = ' '.join(parts[4:]).rsplit(' ', 1)
                processor = processor_until[0]
                until = processor_until[1] if len(processor_until) > 1 else ""
                processes.append({
                    "name": parts[0],
                    "id": parts[1],
                    "size": size,
                    "processor": processor,
                    "until": until
                })
            return processes
        else:
            return []
    except Exception as e:
        print(f"Failed to get running models: {e}")
        return []

def format_gpu_indices(gpu_indices):
    if isinstance(gpu_indices, list):
        return gpu_indices
    elif isinstance(gpu_indices, int):
        return [gpu_indices]
    else:
        return []

def get_os_info():
    os_info = {}
    try:
        with open('/etc/os-release') as f:
            for line in f:
                if '=' in line:
                    key, value = line.rstrip().split('=', 1)
                    os_info[key] = value.strip('"')
    except Exception as e:
        print(f"Failed to read /etc/os-release: {e}")
    return {
        "os_name": os_info.get('NAME', 'Unknown'),
        "os_version": os_info.get('VERSION', 'Unknown')
    }

def get_host_info():
    os_info = get_os_info()
    return {
        "hostname": socket.gethostname(),
        "cpu_load": f"{psutil.cpu_percent(interval=1)}%",
        "os_type": os_info.get("os_name"),
        "os_version": os_info.get("os_version"),
        "memory_utilization": f"{psutil.virtual_memory().percent}%",
        "cpu_cores": psutil.cpu_count(logical=True),
        "swap_utilization": f"{psutil.swap_memory().percent}%"
    }

@app.route('/gpu-info', methods=['GET'])
def gpu_info():
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_info_list = [get_gpu_info(i) for i in range(device_count) if get_gpu_info(i) is not None]
        response = {
            "host": get_host_info(),
            "gpus": gpu_info_list
        }
        pretty_json = json.dumps(response, indent=4)
        return Response(pretty_json, mimetype='application/json')
    except pynvml.NVMLError as e:
        print(f"Failed to get device count: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200


@app.route('/ollama-info', methods=['GET'])
def ollama_info():
    ollama_services = []
    for service in config["ollama_services"]:
        version = get_ollama_version(config["ollama_command"], service["port"])
        models_avail = get_ollama_list(config["ollama_command"], service["port"])
        
        # Get additional information for each model
        for model in models_avail:
            model_details = get_ollama_show(config["ollama_command"], service["port"], model["name"])
            model.update(model_details)
        
        models_running = get_ollama_ps(config["ollama_command"], service["port"])
        service_info = {
            "service_name": service["service_name"],
            "description": service.get("description", ""),
            "port": service["port"],
            "url": service["url"],
            "gpu_indices": format_gpu_indices(service["gpu_indices"]),
            "priority": service["priority"],
            "ollama_version": version,
            "OLLAMA_KEEP_ALIVE": service.get("OLLAMA_KEEP_ALIVE", ""),
            "OLLAMA_MAX_QUEUE": service.get("OLLAMA_MAX_QUEUE", ""),
            "OLLAMA_NUM_PARALLEL": service.get("OLLAMA_NUM_PARALLEL", ""),
            "OLLAMA_MAX_LOADED_MODELS": service.get("OLLAMA_MAX_LOADED_MODELS", ""),
            "OLLAMA_MODELS": service.get("OLLAMA_MODELS", ""),
            "models_avail": models_avail,
            "models_running": models_running
        }
        ollama_services.append(service_info)
    
    other_models = []
    for model in config.get("other_models", []):
        model_info = {
            "service_name": model["service_name"],
            "description": model.get("description", ""),
            "port": model["port"],
            "url": model["url"],
            "gpu_indices": format_gpu_indices(model["gpu_indices"])
        }
        other_models.append(model_info)
    
    ollama_info_response = {
        "node_name": config["node_name"],
        "node_alias": config["node_alias"],
        "node_description": config["node_description"],
        "ollama_command": config["ollama_command"],
        "admin_email": config["admin_email"],
        "ollama_services": ollama_services,
        "other_models": other_models
    }
    pretty_json = json.dumps(ollama_info_response, indent=4)
    return Response(pretty_json, mimetype='application/json')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

