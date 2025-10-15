import unittest
import socket
import time
from scripts.spawn_vllm_server import VLLMServerManager

class TestVLLMServer(unittest.TestCase):
    def test_vllm_server_download_and_serve(self):
        # Use the HF repo path, vllm should handle download
        model_path = "NousResearch/Meta-Llama-3-8B-Instruct"
        port = 8999
        manager = VLLMServerManager(model_path, port)
        manager.start()
        # Wait indefinitely for server to start
        while not manager.is_running():
            time.sleep(2)
        # Check if port is open
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                pass
        except Exception:
            self.fail("vllm server did not open port")
        # Optionally, check /v1/models endpoint
        import requests
        try:
            resp = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=5)
            self.assertEqual(resp.status_code, 200)
            self.assertIn("id", resp.json()["object"])
        finally:
            manager.stop()

if __name__ == "__main__":
    unittest.main()
