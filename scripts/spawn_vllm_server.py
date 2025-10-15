import subprocess
import time
import socket

class VLLMServerManager:
    def __init__(self, model_path, port, host="127.0.0.1"):
        self.model_path = model_path
        self.port = port
        self.host = host
        self.process = None

    def start(self):
        cmd = [
            "vllm", "serve", self.model_path,
            "--host", self.host,
            "--port", str(self.port)
        ]
        self.process = subprocess.Popen(cmd)
        # Wait for server to start
        for _ in range(30):
            if self.is_running():
                return True
            time.sleep(1)
        return False

    def is_running(self):
        try:
            with socket.create_connection((self.host, self.port), timeout=2):
                return True
        except Exception:
            return False

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None

# Example usage:
# manager = VLLMServerManager("models/llama-3-8B.Q4_K_M.gguf", 8000)
# manager.start()
# ...
# manager.stop()
