import unittest

import requests
import time
from scripts.spawn_vllm_server import VLLMServerManager

class TestVLLMClient(unittest.TestCase):
    def setUp(self):
        self.port = 8999
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.model_path = "openai/gpt-oss-20b"
        self.manager = VLLMServerManager(self.model_path, self.port)
        self.manager.start()
        # Wait for server to be up
        for _ in range(30):
            try:
                resp = requests.get(f"{self.base_url}/v1/models", timeout=2)
                if resp.status_code == 200:
                    return
            except Exception:
                time.sleep(1)
        self.manager.stop()
        raise RuntimeError("vLLM server not available")
    def tearDown(self):
        self.manager.stop()

    def test_chat_completion(self):
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": "openai/gpt-oss-20b",
            "messages": [
                {"role": "user", "content": "Hello, who are you?"}
            ],
            "max_tokens": 32
        }
        resp = requests.post(url, json=payload, timeout=10)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("choices", data)
        self.assertGreater(len(data["choices"]), 0)
        self.assertIn("message", data["choices"][0])
        self.assertIn("content", data["choices"][0]["message"])
        print("Model response:", data["choices"][0]["message"]["content"])

if __name__ == "__main__":
    unittest.main()
