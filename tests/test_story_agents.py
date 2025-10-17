import unittest
from scripts.spawn_vllm_server import VLLMServerManager
import requests
import time


STORYTELLER_SYSTEM_PROMPT = (
    "You are a creative storyteller. Write a brief story introduction with up to 4 major characters. You MUST introduce at least one major character in bold (using **like this**), e.g., **Elder Marrow**. Only major characters should be bolded. Your response must be a story with at least one bolded major character."
)
DIRECTOR_SYSTEM_PROMPT = (
    "You are a story director. Only spawn a character agent for major characters introduced in bold (using **like this**) in the story. If there is a new major character, describe their role, personality, and provide a system prompt for the character agent. If there are multiple major characters, output a JSON array, with one object per character, in the following format: [{\"spawn\": true, \"character_name\": string, \"character_prompt\": string}, ...]. If no character should be spawned, output an empty array: []. Respond ONLY with valid JSON, with double quotes, and do not include any other text, explanation, or formatting. Do NOT use the 'reasoning_content' field. Your response MUST be valid JSON in the 'content' field only. Example: [{\"spawn\": true, \"character_name\": \"Elder Marrow\", \"character_prompt\": \"You are Elder Marrow, a wise old shopkeeper with a mysterious past. Respond in character.\"}]"
)

class TestStoryAgents(unittest.TestCase):
    # Removed per user request
    # Removed per user request
    # Removed per user request
    def setUp(self):
        self.model_path = "openai/gpt-oss-20b"
        # Assign ports and GPUs for each agent
        self.storyteller_port = 8999
        self.director_port = 9000
        self.character_port = 9001
        self.storyteller_url = f"http://127.0.0.1:{self.storyteller_port}"
        self.director_url = f"http://127.0.0.1:{self.director_port}"
        self.character_url = f"http://127.0.0.1:{self.character_port}"
        # Start a separate server for each agent, on a different GPU
        self.storyteller_manager = VLLMServerManager(self.model_path, self.storyteller_port, gpu=0, log_file="storyteller_server.log")
        self.director_manager = VLLMServerManager(self.model_path, self.director_port, gpu=1, log_file="director_server.log")
        self.character_manager = VLLMServerManager(self.model_path, self.character_port, gpu=2, log_file="character_server.log")
        self.storyteller_manager.start()
        self.director_manager.start()
        self.character_manager.start()
        # Wait for all servers to be up
        for url, manager in [
            (self.storyteller_url, self.storyteller_manager),
            (self.director_url, self.director_manager),
            (self.character_url, self.character_manager)
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
            self.model_path = "openai/gpt-oss-20b"
            # Assign ports and GPUs for each agent
            self.storyteller_port = 8999
            self.director_port = 9000
            self.character_port = 9001
            self.storyteller_url = f"http://127.0.0.1:{self.storyteller_port}"
            self.director_url = f"http://127.0.0.1:{self.director_port}"
            self.character_url = f"http://127.0.0.1:{self.character_port}"
            # Start a separate server for each agent, on a different GPU
            self.storyteller_manager = VLLMServerManager(self.model_path, self.storyteller_port, gpu=0)
            self.director_manager = VLLMServerManager(self.model_path, self.director_port, gpu=1)
            self.character_manager = VLLMServerManager(self.model_path, self.character_port, gpu=2)
            self.storyteller_manager.start()
            self.director_manager.start()
            self.character_manager.start()
            # Wait for all servers to be up
            for url, manager in [
                (self.storyteller_url, self.storyteller_manager),
                (self.director_url, self.director_manager),
                (self.character_url, self.character_manager)
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

    def tearDown(self):
            self.storyteller_manager.stop()
            self.director_manager.stop()
            self.character_manager.stop()

    def test_three_agent_story(self):
        # 1. Storyteller agent generates a story with one character
        storyteller_prompt = [
            {"role": "system", "content": STORYTELLER_SYSTEM_PROMPT},
            {"role": "user", "content": "Begin the story. Make sure to introduce at least one major character in bold (using **like this**)."}
        ]
        resp = requests.post(
                f"{self.storyteller_url}/v1/chat/completions",
            json={
                "model": "openai/gpt-oss-20b",
                "messages": storyteller_prompt,
                "max_tokens": 512
            }
        )
        self.assertEqual(resp.status_code, 200)
        story = resp.json()["choices"][0]["message"].get("content")
        print("Storyteller agent output:\n", story)
        if not story:
            self.fail("Storyteller agent returned no response.")
        # Ensure at least one bolded character is present
        import re
        bolded = re.findall(r"\*\*(.+?)\*\*", story)
        if not bolded:
            self.fail("Storyteller did not introduce any major character in bold.")

        # 2. Director agent decides if/when to spawn a character agent
        director_prompt = [
            {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": f"Given the following story, spawn a character agent if appropriate. Only output valid JSON in your response. Do not provide any explanation. Story: {story}"}
        ]
        resp_dir = requests.post(
                f"{self.director_url}/v1/chat/completions",
            json={
                "model": "openai/gpt-oss-20b",
                "messages": director_prompt,
                "max_tokens": 2048
            }
        )
        self.assertEqual(resp_dir.status_code, 200)
        print("Full director API response:", resp_dir.json())
        director_reply = resp_dir.json()["choices"][0]["message"].get("content")
        print("Director agent output:\n", director_reply)

        # Fallback: retry once if response is empty
        import json
        if not director_reply:
            print("Director agent returned no response, retrying...")
            resp_dir = requests.post(
                    f"{self.director_url}/v1/chat/completions",
                json={
                    "model": "openai/gpt-oss-20b",
                    "messages": director_prompt,
                    "max_tokens": 2048
                }
            )
            director_reply = resp_dir.json()["choices"][0]["message"].get("content")
            print("Director agent retry output:\n", director_reply)
            if not director_reply:
                self.fail("Director agent returned no response after retry.")
        try:
            director_data = json.loads(director_reply)
        except Exception:
            print("Raw director reply:", director_reply)
            self.fail("Director agent did not return valid JSON.")
        self.assertIsInstance(director_data, list, "Director response should be a JSON array.")
        self.assertGreater(len(director_data), 0, "Director did not spawn any character agents.")
        # Use the first character for the next step
        character_info = director_data[0]
        self.assertTrue(character_info.get("spawn"), "Director did not spawn a character agent.")
        character_name = character_info.get("character_name", "Unknown")
        character_system_prompt = character_info.get("character_prompt", "You are a character.")

        # 3. Character agent responds to the story
        character_prompt = [
            {"role": "system", "content": character_system_prompt},
            {"role": "user", "content": story}
        ]
        resp_char = requests.post(
                f"{self.character_url}/v1/chat/completions",
            json={
                "model": "openai/gpt-oss-20b",
                "messages": character_prompt,
                "max_tokens": 256
            }
        )
        self.assertEqual(resp_char.status_code, 200)
        character_reply = resp_char.json()["choices"][0]["message"]["content"]
        print(f"Character agent ({character_name}) output:\n", character_reply)

if __name__ == "__main__":
    unittest.main()
