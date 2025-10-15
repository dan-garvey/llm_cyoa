import random
from scripts.spawn_vllm_server import VLLMServerManager
import time

# Utility to spawn N vllm servers and return their URLs

def spawn_vllm_servers(model_path, num_servers, base_port=8000):
    managers = []
    urls = []
    for i in range(num_servers):
        port = base_port + i
        manager = VLLMServerManager(model_path, port)
        started = manager.start()
        if not started:
            raise RuntimeError(f"Failed to start vllm server on port {port}")
        managers.append(manager)
        urls.append(f"http://127.0.0.1:{port}")
        time.sleep(2)  # Give server time to warm up
    return managers, urls

# Usage example:
# managers, urls = spawn_vllm_servers("models/llama-3-8B.Q4_K_M.gguf", 3)
# ...
# for m in managers: m.stop()
