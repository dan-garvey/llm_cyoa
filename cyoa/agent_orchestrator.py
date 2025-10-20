STORYTELLER_SYSTEM_PROMPT = (
    "You are a creative storyteller. Write a brief story introduction with up to 4 major characters. You MUST introduce at least one major character in bold (using **like this**), e.g., **Elder Marrow**. Only major characters should be bolded. Your response must be a story with at least one bolded major character."
)
DIRECTOR_SYSTEM_PROMPT = (
    "You are a story director. Only spawn a character agent for major characters introduced in bold (using **like this**) in the story. If there is a new major character, describe their role, personality, and provide a system prompt for the character agent. If there are multiple major characters, output a JSON array, with one object per character, in the following format: [{\"spawn\": true, \"character_name\": string, \"character_prompt\": string}, ...]. If no character should be spawned, output an empty array: []. Respond ONLY with valid JSON, with double quotes, and do not include any other text, explanation, or formatting. Do NOT use the 'reasoning_content' field. Your response MUST be valid JSON in the 'content' field only. Example: [{\"spawn\": true, \"character_name\": \"Elder Marrow\", \"character_prompt\": \"You are Elder Marrow, a wise old shopkeeper with a mysterious past. Respond in character.\"}]"
)
def get_user_character_info():
    """Prompt the user for character name and background info."""
    import sys
    print("Enter your character's name:", end=' ', flush=True)
    name = sys.stdin.readline().strip()
    print("Enter any background info for your character (optional):", end=' ', flush=True)
    background = sys.stdin.readline().strip()
    return name, background
import time
import requests
from scripts.spawn_vllm_server import VLLMServerManager

class AgentOrchestrator:
    # ANSI color codes for log coloring
    AGENT_COLORS = {
        'Storyteller': '\033[95m',  # Magenta
        'Director': '\033[94m',     # Blue
        'Character': '\033[92m',    # Green
        'ENDC': '\033[0m'
    }

    def set_logger(self, logger):
        self.logger = logger

    def log_agent(self, agent_type, message, prompt_or_response, agent_name=None):
        color = self.AGENT_COLORS.get(agent_type, '')
        endc = self.AGENT_COLORS['ENDC']
        name_str = f" ({agent_name})" if agent_name else ""
        log_msg = f"{color}[{agent_type}{name_str}] {message}:{endc}\n{prompt_or_response}\n"
        if hasattr(self, 'logger') and self.logger:
            self.logger.debug(log_msg)
        else:
            print(log_msg)
    def build_character_prompt(self, character_name, character_system_prompt, visible_story_segment):
        return [
            {"role": "system", "content": (
                f"You are {character_name}. Respond in character to the following events in the world. "
                "Your response will be sent to the director, who will integrate it into the ongoing story. "
                "Only respond to what you can see or hear. If you are not present in the scene, respond with an empty string."
            )},
            {"role": "system", "content": character_system_prompt},
            {"role": "user", "content": visible_story_segment}
        ]

    def director_distribute_and_collect(self, story, director_data, user_name):
        """
        For each character agent (not the user), send the relevant story segment and collect their responses.
        Returns: dict mapping character_name -> response
        """
        responses = {}
        for char in director_data:
            if not char.get("spawn"):
                continue
            character_name = char.get("character_name")
            if character_name == user_name:
                continue  # Never spawn agent for user
            character_system_prompt = char.get("character_prompt", "You are a character.")
            # For now, send the whole story to each agent; in a real system, parse for relevant segments
            character_prompt = self.build_character_prompt(character_name, character_system_prompt, story)
            self.start_character_manager()  # (Re)start for each agent; in a real system, pool/reuse
            resp_char = self.post_with_retries(
                f"{self.character_url}/v1/chat/completions",
                {
                    "model": self.model_path,
                    "messages": character_prompt,
                    "max_tokens": 256
                }
            )
            if resp_char.status_code == 200:
                char_reply = resp_char.json()["choices"][0]["message"]["content"]
            else:
                char_reply = ""
            responses[character_name] = char_reply
        return responses

    def director_integrate_character_responses(self, story, char_responses):
        """
        Integrate character agent responses into the story. For now, append each response to the story.
        In a real system, this would be more sophisticated.
        """
        for char, reply in char_responses.items():
            if reply.strip():
                story += f"\n[{char}]: {reply.strip()}"
        return story

    def interactive_story_loop(self, user_name, user_background, user_inputs, max_turns=5):
        """
        Main loop: storyteller -> director -> character agents -> director integrates -> user.
        user_inputs: list of user input strings for each turn.
        """
        story = None
        # If no user input, generate and return the story introduction only
        if not user_inputs:
            storyteller_prompt = self.build_storyteller_prompt_with_user(user_name, user_background)
            self.log_agent('Storyteller', 'Prompt', storyteller_prompt)
            resp = self.post_with_retries(
                f"{self.storyteller_url}/v1/chat/completions",
                {
                    "model": self.model_path,
                    "messages": storyteller_prompt,
                    "max_tokens": 512
                }
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Storyteller agent failed: {resp.status_code}")
            story = resp.json()["choices"][0]["message"].get("content")
            self.log_agent('Storyteller', 'Response', story)
            return story

        for turn, user_input in enumerate(user_inputs[:max_turns]):
            # 1. Storyteller generates next story segment
            if turn == 0:
                storyteller_prompt = self.build_storyteller_prompt_with_user(user_name, user_background)
                self.log_agent('Storyteller', 'Prompt', storyteller_prompt)
            else:
                storyteller_prompt = [
                    {"role": "system", "content": f"Continue the story. The user says: {user_input}"},
                    {"role": "system", "content": STORYTELLER_SYSTEM_PROMPT},
                    {"role": "user", "content": story}
                ]
                self.log_agent('Storyteller', 'Prompt', storyteller_prompt)
            resp = self.post_with_retries(
                f"{self.storyteller_url}/v1/chat/completions",
                {
                    "model": self.model_path,
                    "messages": storyteller_prompt,
                    "max_tokens": 512
                }
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Storyteller agent failed: {resp.status_code}")
            story = resp.json()["choices"][0]["message"].get("content")
            self.log_agent('Storyteller', 'Response', story)

            # 2. Director decides which character agents to spawn (excluding user)
            director_prompt = self.build_director_prompt(story, user_name)
            self.log_agent('Director', 'Prompt', director_prompt)
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
            import json
            director_reply = resp_dir.json()["choices"][0]["message"].get("content")
            self.log_agent('Director', 'Response', director_reply)
            director_data = json.loads(director_reply)

            # 3. Director distributes story to character agents and collects responses
            char_responses = {}
            for char in director_data:
                if not char.get("spawn"):
                    continue
                character_name = char.get("character_name")
                if character_name == user_name:
                    continue  # Never spawn agent for user
                character_system_prompt = char.get("character_prompt", "You are a character.")
                character_prompt = self.build_character_prompt(character_name, character_system_prompt, story)
                self.log_agent('Character', 'Prompt', character_prompt, agent_name=character_name)
                self.start_character_manager()
                resp_char = self.post_with_retries(
                    f"{self.character_url}/v1/chat/completions",
                    {
                        "model": self.model_path,
                        "messages": character_prompt,
                        "max_tokens": 256
                    }
                )
                if resp_char.status_code == 200:
                    char_reply = resp_char.json()["choices"][0]["message"]["content"]
                else:
                    char_reply = ""
                self.log_agent('Character', 'Response', char_reply, agent_name=character_name)
                char_responses[character_name] = char_reply

            # 4. Director integrates character responses into the story
            story = self.director_integrate_character_responses(story, char_responses)

            # 5. Present updated story to user (in a real app, yield/return here)
                # Only return story; main_app.py handles user-facing output
        return story
    def build_storyteller_prompt_with_user(self, user_name, user_background):
        explanation = (
            f"The story should primarily be about the user character: **{user_name}**. "
            "Use the following background information to inform the story. "
            "Make sure to introduce this character in bold (using **like this**), and center the story around them. "
            "Background: " + (user_background if user_background else "None provided.")
        )
        return [
            {"role": "system", "content": explanation},
            {"role": "system", "content": STORYTELLER_SYSTEM_PROMPT},
            {"role": "user", "content": "Begin the story. Make sure to introduce at least one major character in bold (using **like this**)."}
        ]

    def build_director_prompt(self, story, user_name):
        return [
            {"role": "system", "content": (
                f"You are a story director. The main character is {user_name}. Do NOT spawn an agent for this character, but treat them as the user. "
                "Only spawn a character agent for other major characters introduced in bold (using **like this**) in the story. "
                "If there is a new major character (not the user), describe their role, personality, and provide a system prompt for the character agent. "
                "If there are multiple major characters, output a JSON array, with one object per character, in the following format: [{\"spawn\": true, \"character_name\": string, \"character_prompt\": string}, ...]. "
                "If no character should be spawned, output an empty array: []. Respond ONLY with valid JSON, with double quotes, and do not include any other text, explanation, or formatting. "
                "Do NOT use the 'reasoning_content' field. Your response MUST be valid JSON in the 'content' field only. "
                "Example: [{\"spawn\": true, \"character_name\": \"Elder Marrow\", \"character_prompt\": \"You are Elder Marrow, a wise old shopkeeper with a mysterious past. Respond in character.\"}]"
            )},
            {"role": "user", "content": f"Given the following story, spawn a character agent if appropriate (but never for {user_name}). Only output valid JSON in your response. Do not provide any explanation. Story: {story}"}
        ]
    def __init__(self, model_path, storyteller_port=8999, director_port=9000, character_port=9001, storyteller_gpu=0, director_gpu=1, character_gpu=2):
        self.model_path = model_path
        self.storyteller_port = storyteller_port
        self.director_port = director_port
        self.character_port = character_port
        self.storyteller_gpu = storyteller_gpu
        self.director_gpu = director_gpu
        self.character_gpu = character_gpu
        self.storyteller_url = f"http://127.0.0.1:{self.storyteller_port}"
        self.director_url = f"http://127.0.0.1:{self.director_port}"
        self.character_url = f"http://127.0.0.1:{self.character_port}"
        self.storyteller_manager = VLLMServerManager(self.model_path, self.storyteller_port, gpu=self.storyteller_gpu, log_file="storyteller_server.log")
        self.director_manager = VLLMServerManager(self.model_path, self.director_port, gpu=self.director_gpu, log_file="director_server.log")
        self.character_manager = None

    def wait_for_server_ready(self, url, timeout=60):
        import requests
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = requests.get(f"{url}/v1/models", timeout=2)
                if resp.status_code == 200:
                    return True
            except Exception:
                time.sleep(1)
        raise RuntimeError(f"vLLM server not available at {url}")

    def start_storyteller_and_director(self):
        self.storyteller_manager.start()
        self.director_manager.start()

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
        self.character_manager = VLLMServerManager(self.model_path, self.character_port, gpu=self.character_gpu, log_file="character_server.log")
        self.character_manager.start()

    def post_with_retries(self, url, payload, max_retries=5, wait=3):
        import requests
        for attempt in range(max_retries):
            try:
                resp = requests.post(url, json=payload)
            except requests.exceptions.ConnectionError:
                # Wait for server if connection fails
                # Example: url = 'http://127.0.0.1:8999/v1/chat/completions'
                # url.rsplit('/v1', 1)[0] -> 'http://127.0.0.1:8999'
                self.wait_for_server_ready(url.rsplit('/v1', 1)[0])
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
