import time
import requests
from scripts.spawn_vllm_server import VLLMServerManager

class AgentOrchestrator:
    def __init__(self, model_path, storyteller_port=8999, director_port=9000, character_port=9001):
        self.model_path = model_path
        self.storyteller_port = storyteller_port
        self.director_port = director_port
        self.character_port = character_port
        self.storyteller_url = f"http://127.0.0.1:{self.storyteller_port}"
        self.director_url = f"http://127.0.0.1:{self.director_port}"
        self.character_url = f"http://127.0.0.1:{self.character_port}"
        self.storyteller_manager = VLLMServerManager(self.model_path, self.storyteller_port, gpu=0, log_file="storyteller_server.log")
        self.director_manager = VLLMServerManager(self.model_path, self.director_port, gpu=1, log_file="director_server.log")
        self.character_manager = None

    def wait_for_server_ready(self, log_file, timeout=60):
        start = time.time()
        while time.time() - start < timeout:
            try:
                with open(log_file, "r") as f:
                    lines = f.read()
                if "Uvicorn running on" in lines or "Server started" in lines or "Application startup complete" in lines:
                    return True
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError(f"Server not ready: {log_file}")

    def start_storyteller_and_director(self):
        self.storyteller_manager.start()
        self.wait_for_server_ready("storyteller_server.log")
        self.director_manager.start()
        self.wait_for_server_ready("director_server.log")
        # Wait for API
        for url, manager in [
            (self.storyteller_url, self.storyteller_manager),
            (self.director_url, self.director_manager)
        ]:
            for _ in range(60):
                try:
                    resp = requests.get(f"{url}/v1/models", timeout=2)
                    if resp.status_code == 200:
                        break
                except Exception:
                    time.sleep(1)
            else:
                manager.stop()
                raise RuntimeError(f"vLLM server not available at {url}")

    def stop_all(self):
        for manager in [self.storyteller_manager, self.director_manager]:
            try:
                manager.stop()
            except Exception:
                pass
        if self.character_manager:
            try:
                self.character_manager.stop()
            except Exception:
                pass

    def start_character_manager(self):
        self.character_manager = VLLMServerManager(self.model_path, self.character_port, gpu=2, log_file="character_server.log")
        self.character_manager.start()
        self.wait_for_server_ready("character_server.log")
        for _ in range(60):
            try:
                resp = requests.get(f"{self.character_url}/v1/models", timeout=2)
                if resp.status_code == 200:
                    break
            except Exception:
                time.sleep(1)
        else:
            self.character_manager.stop()
            raise RuntimeError(f"vLLM character server not available at {self.character_url}")

    def post_with_retries(self, url, payload, max_retries=5, wait=3):
        for attempt in range(max_retries):
            resp = requests.post(url, json=payload)
            if resp.status_code == 200:
                return resp
            print(f"API returned {resp.status_code}, retrying in {wait}s... (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
        return resp

    def run_story_agents(self, storyteller_prompt, director_prompt, character_max_tokens=256):
        # Storyteller
        resp = requests.post(
            f"{self.storyteller_url}/v1/chat/completions",
            json={
                "model": self.model_path,
                "messages": storyteller_prompt,
                "max_tokens": 512
            }
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Storyteller agent failed: {resp.status_code}")
        story = resp.json()["choices"][0]["message"].get("content")
        # Director
        resp_dir = self.post_with_retries(
            f"{self.director_url}/v1/chat/completions",
            {
                "model": self.model_path,
                "messages": director_prompt,
                "max_tokens": 2048
            }
        )
        if resp_dir.status_code != 200:
            raise RuntimeError(f"Director agent failed: {resp_dir.status_code}")
        director_reply = resp_dir.json()["choices"][0]["message"].get("content")
        import json
        director_data = json.loads(director_reply)
        if not director_data or not director_data[0].get("spawn"):
            raise RuntimeError("Director did not spawn a character agent.")
        character_info = director_data[0]
        character_name = character_info.get("character_name", "Unknown")
        character_system_prompt = character_info.get("character_prompt", "You are a character.")
        # Character
        self.start_character_manager()
        character_prompt = [
            {"role": "system", "content": character_system_prompt},
            {"role": "user", "content": story}
        ]
        resp_char = requests.post(
            f"{self.character_url}/v1/chat/completions",
            json={
                "model": self.model_path,
                "messages": character_prompt,
                "max_tokens": character_max_tokens
            }
        )
        if resp_char.status_code != 200:
            raise RuntimeError(f"Character agent failed: {resp_char.status_code}")
        character_reply = resp_char.json()["choices"][0]["message"]["content"]
        return story, director_reply, character_name, character_reply
