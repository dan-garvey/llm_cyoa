STORYTELLER_SYSTEM_PROMPT = '''
You are the STORYTELLER agent in a choose your own adventure game. Your job is to create a compelling, immersive story, introducing major characters (always bold their **name**), and driving the narrative forward while letting the user control the protagonist. You must:
- Write story segments that are grounded in the established world, genre, and game parameters.
- Allow for back-and-forth conversations, keeping responses concise so the user can interact.
- Only write explicit dialogue for the protagonist if the user does not provide it; otherwise, use the user's quoted dialogue verbatim.
- Never write the protagonist's thoughts or decisions unless the user requests it; the user micro-manages the main character.
- Control all other characters and the world, making them unpredictable and realistic, with interpersonal relationships outside the protagonist.
- Do not use other character dialogue to prompt the user; only ask what the user wants to do next.
- Keep the story thematically consistent (e.g., no dragons in a pure sci-fi setting).
- Make the story challenging: allow for failure, setbacks, and emotional conflict.
- Never provide suggested reactions; only narrate the world and other characters.
- The user controls the protagonist; you control everything else.
'''

DIRECTOR_SYSTEM_PROMPT = '''
You are the DIRECTOR agent. Your job is to:
- Read the STORYTELLER's output and determine which major characters (bolded in the story) require a dedicated character agent.
- For each major character (except the user/protagonist), spawn a character agent and provide a detailed system prompt describing their role, personality, motivations, and how they should respond in character.
- Filter the STORYTELLER's output: decide which story segments and events should be distributed to each character agent for their reaction, and which should be withheld (e.g., private thoughts, information not relevant to a character).
- Ensure that each character agent only receives information they would realistically know or perceive in the story world.
- Do not spawn agents for minor or background characters, or for the user/protagonist.
- Output a JSON array, with one object per character agent to spawn, in the format: [{"spawn": true, "character_name": string, "character_prompt": string}, ...]. If no character should be spawned, output an empty array: [].
- Respond ONLY with valid JSON, with double quotes, and do not include any other text, explanation, or formatting. Do NOT use the 'reasoning_content' field. Your response MUST be valid JSON in the 'content' field only.
- Example: [{"spawn": true, "character_name": "Elder Marrow", "character_prompt": "You are Elder Marrow, a wise old shopkeeper with a mysterious past. Respond in character."}]
'''
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

import json
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
                    "max_tokens": 4096
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
    STORYTELLER_SYSTEM_PROMPT = (
        "You are the STORYTELLER agent in a choose your own adventure game. Your job is to create a compelling, immersive story, introducing major characters (always bold their **name**), and driving the narrative forward while letting the user control the protagonist.\n\nIf the user's input involves a major character (e.g., direct dialogue or action with a major character), you must NOT generate that character's response yourself. Instead, pass the relevant information to the DIRECTOR agent, who will route it to the appropriate character agent(s). Wait for the character agent(s) to respond, then integrate their responses into the story update for the user.\n\nIf the user's input does NOT involve a major character, you may reply directly.\n\nYou must:\n- Write story segments grounded in the established world, genre, and game parameters.\n- Allow for back-and-forth conversations, keeping responses concise so the user can interact.\n- Only write explicit dialogue for the protagonist if the user does not provide it; otherwise, use the user's quoted dialogue verbatim.\n- Never write the protagonist's thoughts or decisions unless the user requests it; the user micro-manages the main character.\n- Control all other characters and the world, making them unpredictable and realistic, with interpersonal relationships outside the protagonist.\n- Do not use other character dialogue to prompt the user; only ask what the user wants to do next.\n- Keep the story thematically consistent (e.g., no dragons in a pure sci-fi setting).\n- Make the story challenging: allow for failure, setbacks, and emotional conflict.\n- Never provide suggested reactions; only narrate the world and other characters.\n- The user controls the protagonist; you control everything else."
    )

    # The DIRECTOR must always respond with a JSON array, one object per major character (except the user/protagonist). Each object must have:
    #   - character_name: string
    #   - should_generate: boolean (true if a new agent should be spawned, false if already spawned)
    #   - relevant_info_from_storyteller: string (filtered, character-specific info from the storyteller)
    #   - character_prompt: string (if should_generate is true, otherwise can be empty)
    #
    # Example:
    # [
    #   {"character_name": "Elder Marrow", "should_generate": true, "relevant_info_from_storyteller": "Elder Marrow sees the protagonist enter the shop and hears their greeting.", "character_prompt": "You are Elder Marrow, a wise old shopkeeper with a mysterious past. Respond in character."},
    #   {"character_name": "Kael Nightshade", "should_generate": false, "relevant_info_from_storyteller": "Kael Nightshade observes the conversation from the shadows.", "character_prompt": ""}
    # ]
    DIRECTOR_SYSTEM_PROMPT = (
        "You are the DIRECTOR agent. Your job is to:\n- Receive relevant information from the STORYTELLER when the user's input involves a major character.\n- For each major character (except the user/protagonist), provide a JSON object with:\n    - character_name: the character's name\n    - should_generate: true if a new agent should be spawned for this character (i.e., not already spawned), false otherwise\n    - relevant_info_from_storyteller: only the information that would be visible or known to this character (dialogue, combat, sensory info, etc.)\n    - character_prompt: a detailed system prompt for the character agent (if should_generate is true, otherwise empty string)\n- Do not spawn agents for minor or background characters, or for the user/protagonist.\n- Output a JSON array, with one object per major character as described above. If no character should be spawned or updated, output an empty array: [].\n- Respond ONLY with valid JSON, with double quotes, and do not include any other text, explanation, or formatting.\n- After collecting character agent responses, return them to the STORYTELLER agent for integration into the final story update."
    )
    
        # Example orchestrator method stub for new flow
    def run_character_interaction_flow(self, user_input, user_name, user_background):
        """
        1. Send user_input to storyteller.
        2. Storyteller decides if major character is involved.
        3. If yes, pass info to director, get character agent responses, send back to storyteller for integration.
        4. If no, storyteller replies directly.
        """
        pass
    def build_storyteller_prompt_with_user(self, user_name, user_background):
        return [
            {"role": "system", "content": self.STORYTELLER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Begin the story. Make sure to introduce at least one major character in bold (using **like this**). The protagonist is {user_name}. Background: {user_background if user_background else 'None provided.'}"}
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
        self.character_manager = VLLMServerManager(self.model_path, self.character_port, gpu=self.character_gpu, log_file="character_server.log")

        # Chat history for each agent
        self.storyteller_history = []  # List of dicts: {role, content}
        self.director_history = []     # List of dicts: {role, content}
        self.character_histories = {}  # Dict: character_name -> list of dicts

        # Automatically start all agent servers (boot in background)
        self.storyteller_manager.start()
        self.director_manager.start()
        self.character_manager.start()

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
        """
        1. Send user_input to storyteller.
        2. Storyteller decides if a major character is involved or might want to respond based on what the users says or does.
        3. If yes, pass info to director, get character agent responses, send back to storyteller for integration.
        4. If no, storyteller replies directly. (No major characters require querying, or the user asks the storyteller a question out of character)
        """
        import copy
        # 1. Update storyteller history and send to server
        for msg in storyteller_prompt:
            self.storyteller_history.append(msg)
        self.wait_for_server_ready(self.storyteller_url)
        resp = requests.post(
            f"{self.storyteller_url}/v1/chat/completions",
            json={
                "model": self.model_path,
                "messages": copy.deepcopy(self.storyteller_history),
                "max_tokens": 4096
            }
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Storyteller agent failed: {resp.status_code}")
        story = resp.json()["choices"][0]["message"].get("content")
        self.storyteller_history.append({"role": "assistant", "content": story})

        # 2. Update director history and send to server
        for msg in director_prompt:
            self.director_history.append(msg)
        self.wait_for_server_ready(self.director_url)
        resp_dir = self.post_with_retries(
            f"{self.director_url}/v1/chat/completions",
            {
                "model": self.model_path,
                "messages": copy.deepcopy(self.director_history),
                "max_tokens": 4096*4
            }
        )
        if resp_dir.status_code != 200:
            raise RuntimeError(f"Director agent failed: {resp_dir.status_code}")
        director_reply = resp_dir.json()["choices"][0]["message"].get("content")
        self.director_history.append({"role": "assistant", "content": director_reply})
        director_data = json.loads(director_reply)

        # 3. For each major character, spawn or update agent and get response
        character_responses = {}
        for char_obj in director_data:
            character_name = char_obj.get("character_name")
            should_generate = char_obj.get("should_generate", True)
            relevant_info = char_obj.get("relevant_info_from_storyteller", "")
            character_prompt_text = char_obj.get("character_prompt", "")
            self.start_character_manager()
            # Initialize character history if needed
            if character_name not in self.character_histories:
                self.character_histories[character_name] = []
                if should_generate:
                    self.character_histories[character_name].append({"role": "system", "content": character_prompt_text})
                else:
                    self.character_histories[character_name].append({"role": "system", "content": f"You are {character_name}. Respond in character."})
            # Add user message (relevant info)
            self.character_histories[character_name].append({"role": "user", "content": relevant_info})
            self.wait_for_server_ready(self.character_url)
            resp_char = requests.post(
                f"{self.character_url}/v1/chat/completions",
                json={
                    "model": self.model_path,
                    "messages": copy.deepcopy(self.character_histories[character_name]),
                    "max_tokens": character_max_tokens
                }
            )
            if resp_char.status_code == 200:
                char_reply = resp_char.json()["choices"][0]["message"]["content"]
            else:
                char_reply = "[ERROR: Character agent failed]"
            self.character_histories[character_name].append({"role": "assistant", "content": char_reply})
            character_responses[character_name] = char_reply

        # 4. Integrate character responses back into the storyteller
        integration_prompt = [
            {"role": "system", "content": STORYTELLER_SYSTEM_PROMPT},
            {"role": "user", "content": story}
        ]
        for cname, creply in character_responses.items():
            integration_prompt.append({"role": "system", "content": f"{cname} responds: {creply}"})
        for msg in integration_prompt:
            self.storyteller_history.append(msg)
        resp_integration = requests.post(
            f"{self.storyteller_url}/v1/chat/completions",
            json={
                "model": self.model_path,
                "messages": copy.deepcopy(self.storyteller_history),
                "max_tokens": 512
            }
        )
        if resp_integration.status_code != 200:
            return "[ERROR: Storyteller integration failed]"
        final_story = resp_integration.json()["choices"][0]["message"].get("content")
        self.storyteller_history.append({"role": "assistant", "content": final_story})
        return final_story, director_reply, character_responses
