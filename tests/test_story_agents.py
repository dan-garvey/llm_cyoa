import unittest
from cyoa.agent_orchestrator import AgentOrchestrator


STORYTELLER_SYSTEM_PROMPT = (
    "You are a creative storyteller. Write a brief story introduction with up to 4 major characters. You MUST introduce at least one major character in bold (using **like this**), e.g., **Elder Marrow**. Only major characters should be bolded. Your response must be a story with at least one bolded major character."
)
DIRECTOR_SYSTEM_PROMPT = (
    "You are a story director. Only spawn a character agent for major characters introduced in bold (using **like this**) in the story. If there is a new major character, describe their role, personality, and provide a system prompt for the character agent. If there are multiple major characters, output a JSON array, with one object per character, in the following format: [{\"spawn\": true, \"character_name\": string, \"character_prompt\": string}, ...]. If no character should be spawned, output an empty array: []. Respond ONLY with valid JSON, with double quotes, and do not include any other text, explanation, or formatting. Do NOT use the 'reasoning_content' field. Your response MUST be valid JSON in the 'content' field only. Example: [{\"spawn\": true, \"character_name\": \"Elder Marrow\", \"character_prompt\": \"You are Elder Marrow, a wise old shopkeeper with a mysterious past. Respond in character.\"}]"
)

class TestStoryAgents(unittest.TestCase):
    def wait_for_server_ready(self, log_file, timeout=60):
        """Poll the server log file for a 'ready' message before proceeding."""
        import time
        start = time.time()
        while time.time() - start < timeout:
            try:
                with open(log_file, "r") as f:
                    lines = f.read()
                # Look for Uvicorn or vLLM ready message
                if "Uvicorn running on" in lines or "Server started" in lines or "Application startup complete" in lines:
                    return True
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError(f"Server not ready: {log_file}")
    def setUp(self):
        self.model_path = "openai/gpt-oss-20b"
        self.orchestrator = AgentOrchestrator(self.model_path)
        self.orchestrator.start_storyteller_and_director()

    def tearDown(self):
        self.orchestrator.stop_all()

    def test_three_agent_story(self):
        # 1. Storyteller agent generates a story
        storyteller_prompt = [
            {"role": "system", "content": STORYTELLER_SYSTEM_PROMPT},
            {"role": "user", "content": "Begin the story. Make sure to introduce at least one major character in bold (using **like this**)."}
        ]
        resp = self.orchestrator.post_with_retries(
            f"{self.orchestrator.storyteller_url}/v1/chat/completions",
            {
                "model": self.orchestrator.model_path,
                "messages": storyteller_prompt,
                "max_tokens": 512
            }
        )
        self.assertEqual(resp.status_code, 200)
        story = resp.json()["choices"][0]["message"].get("content")
        print("Storyteller agent output:\n", story)
        import re
        bolded = re.findall(r"\*\*(.+?)\*\*", story)
        self.assertTrue(bolded, "Storyteller did not introduce any major character in bold.")

        # 2. Director agent decides if/when to spawn a character agent
        director_prompt = [
            {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": f"Given the following story, spawn a character agent if appropriate. Only output valid JSON in your response. Do not provide any explanation. Story: {story}"}
        ]
        resp_dir = self.orchestrator.post_with_retries(
            f"{self.orchestrator.director_url}/v1/chat/completions",
            {
                "model": self.orchestrator.model_path,
                "messages": director_prompt,
                "max_tokens": 2048
            }
        )
        self.assertEqual(resp_dir.status_code, 200)
        director_reply = resp_dir.json()["choices"][0]["message"].get("content")
        print("Director agent output:\n", director_reply)
        import json
        director_data = json.loads(director_reply)
        self.assertIsInstance(director_data, list, "Director response should be a JSON array.")
        self.assertGreater(len(director_data), 0, "Director did not spawn any character agents.")
        character_info = director_data[0]
        self.assertTrue(character_info.get("spawn"), "Director did not spawn a character agent.")
        character_name = character_info.get("character_name", "Unknown")
        character_system_prompt = character_info.get("character_prompt", "You are a character.")

        # 3. Spawn character agent server only when needed
        self.orchestrator.start_character_manager()
        character_prompt = [
            {"role": "system", "content": character_system_prompt},
            {"role": "user", "content": story}
        ]
        resp_char = self.orchestrator.post_with_retries(
            f"{self.orchestrator.character_url}/v1/chat/completions",
            {
                "model": self.orchestrator.model_path,
                "messages": character_prompt,
                "max_tokens": 256
            }
        )
        self.assertEqual(resp_char.status_code, 200)
        character_reply = resp_char.json()["choices"][0]["message"]["content"]
        print(f"Character agent ({character_name}) output:\n", character_reply)

if __name__ == "__main__":
    unittest.main()
