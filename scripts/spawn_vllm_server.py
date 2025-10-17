import subprocess
import time
import socket

class VLLMServerManager:
    def __init__(self, model_path, port, host="127.0.0.1", gpu=None, log_file=None):
        self.model_path = model_path
        self.port = port
        self.host = host
        self.gpu = gpu
        self.log_file = log_file
        self.process = None

    def start(self):
        import os
        cmd = [
            "vllm", "serve", self.model_path,
            "--host", self.host,
            "--port", str(self.port)
        ]
        env = os.environ.copy()
        if self.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        stdout = None
        stderr = None
        if self.log_file:
            log_fh = open(self.log_file, "a", buffering=1)
            stdout = log_fh
            stderr = log_fh
            self._log_fh = log_fh
        else:
            self._log_fh = None
        self.process = subprocess.Popen(cmd, preexec_fn=os.setpgrp, env=env, stdout=stdout, stderr=stderr, close_fds=True)
        # Wait for server to start
        for _ in range(60):
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
        import os, signal
        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except Exception:
                pass
            self.process.wait(timeout=10)
            self.process = None
        if hasattr(self, '_log_fh') and self._log_fh:
            self._log_fh.close()

# Example usage:
# manager = VLLMServerManager("models/llama-3-8B.Q4_K_M.gguf", 8000)
# manager.start()
# ...
# manager.stop()
