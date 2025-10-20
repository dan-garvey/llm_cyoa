import unittest
from cyoa.agent_orchestrator import AgentOrchestrator


STORYTELLER_SYSTEM_PROMPT = (
    "You are a creative storyteller. Write a brief story introduction with up to 4 major characters. You MUST introduce at least one major character in bold (using **like this**), e.g., **Elder Marrow**. Only major characters should be bolded. Your response must be a story with at least one bolded major character."
)
DIRECTOR_SYSTEM_PROMPT = (
    "You are a story director. Only spawn a character agent for major characters introduced in bold (using **like this**) in the story. If there is a new major character, describe their role, personality, and provide a system prompt for the character agent. If there are multiple major characters, output a JSON array, with one object per character, in the following format: [{\"spawn\": true, \"character_name\": string, \"character_prompt\": string}, ...]. If no character should be spawned, output an empty array: []. Respond ONLY with valid JSON, with double quotes, and do not include any other text, explanation, or formatting. Do NOT use the 'reasoning_content' field. Your response MUST be valid JSON in the 'content' field only. Example: [{\"spawn\": true, \"character_name\": \"Elder Marrow\", \"character_prompt\": \"You are Elder Marrow, a wise old shopkeeper with a mysterious past. Respond in character.\"}]"
)

class TestStoryAgents(unittest.TestCase):
    def test_story_with_user_character(self):
        # Hardcoded user character info
        user_name = "Astra Vey"
        user_background = "A mysterious wanderer with a silver compass, seeking the lost city of mirrors."
        # Build storyteller prompt with user info
        storyteller_prompt = self.orchestrator.build_storyteller_prompt_with_user(user_name, user_background)
        # Only prompt the storyteller agent ONCE, with debug logging
        print("\n[DEBUG] Storyteller prompt:")
        for msg in storyteller_prompt:
            print(f"  {msg['role']}: {msg['content']}")
        resp = self.orchestrator.post_with_retries(
            f"{self.orchestrator.storyteller_url}/v1/chat/completions",
            {
                "model": self.model_path,
                "messages": storyteller_prompt,
                "max_tokens": 512
            }
        )
        self.assertEqual(resp.status_code, 200)
        story = resp.json()["choices"][0]["message"].get("content")
        print("\nStoryteller agent output:\n", story)
        # Check that the user's character is introduced in bold
        import re
        bolded = re.findall(r"\*\*(.+?)\*\*", story)
        self.assertIn(user_name, bolded, f"User character {user_name} not introduced in bold.")
        self.assertIn(user_name, story, f"User character {user_name} not present in story.")
        # Continue with director and character agent as in main test, with debug logging
        director_prompt = self.orchestrator.build_director_prompt(story, user_name)
        print("\n[DEBUG] Director prompt:")
        for msg in director_prompt:
            print(f"  {msg['role']}: {msg['content']}")
        resp_dir = self.orchestrator.post_with_retries(
            f"{self.orchestrator.director_url}/v1/chat/completions",
            {
                "model": self.model_path,
                "messages": director_prompt,
                "max_tokens": 2048
            }
        )
        self.assertEqual(resp_dir.status_code, 200)
        director_reply = resp_dir.json()["choices"][0]["message"].get("content")
        print("\nDirector agent output:\n", director_reply)
        import json
        director_data = json.loads(director_reply)
        self.assertIsInstance(director_data, list, "Director response should be a JSON array.")
        self.assertGreater(len(director_data), 0, "Director did not spawn any character agents.")
        character_info = director_data[0]
        self.assertTrue(character_info.get("spawn"), "Director did not spawn a character agent.")
        character_name = character_info.get("character_name", "Unknown")
        character_system_prompt = character_info.get("character_prompt", "You are a character.")
        self.orchestrator.start_character_manager()
        character_prompt = [
            {"role": "system", "content": character_system_prompt},
            {"role": "user", "content": story}
        ]
        print("\n[DEBUG] Character prompt:")
        for msg in character_prompt:
            print(f"  {msg['role']}: {msg['content']}")
        resp_char = self.orchestrator.post_with_retries(
            f"{self.orchestrator.character_url}/v1/chat/completions",
            {
                "model": self.model_path,
                "messages": character_prompt,
                "max_tokens": 256
            }
        )
        self.assertEqual(resp_char.status_code, 200)
        character_reply = resp_char.json()["choices"][0]["message"]["content"]
        print(f"\nCharacter agent ({character_name}) output:\n", character_reply)
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
        self.model_path = "meta-llama/Llama-3.2-3B-Instruct"
        # Assign GPUs explicitly (change as needed for your system)
        self.orchestrator = AgentOrchestrator(
            self.model_path,
            storyteller_gpu=0,
            director_gpu=1,
            character_gpu=2
        )
        self.orchestrator.start_storyteller_and_director()

    def tearDown(self):
        self.orchestrator.stop_all()


if __name__ == "__main__":
    unittest.main()
