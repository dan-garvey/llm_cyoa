

import requests

class OverallAgent:
    def __init__(self, server_url):
        self.server_url = server_url
        self.max_tokens = 256

    def generate_response(self, prompt, character_agents):
        char_names = ', '.join([agent.name for agent in character_agents])
        messages = [
            {"role": "system", "content": f"You are the narrator of a text adventure. Characters: {char_names}. Respond with dialogue for each character."},
            {"role": "user", "content": prompt}
        ]
        payload = {
            "messages": messages,
            "max_tokens": self.max_tokens
        }
        response = requests.post(f"{self.server_url}/v1/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class CharacterAgent:
    def __init__(self, name, server_url):
        self.name = name
        self.server_url = server_url
        self.max_tokens = 64

    def generate_dialogue(self, context):
        messages = [
            {"role": "system", "content": f"You are {self.name}, a character in a text adventure. Respond in character."},
            {"role": "user", "content": context}
        ]
        payload = {
            "messages": messages,
            "max_tokens": self.max_tokens
        }
        response = requests.post(f"{self.server_url}/v1/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]



class ResponseAllocator:
    def allocate(self, overall_response, character_agents):
        lines = overall_response.split('\n')
        allocations = {}
        for agent in character_agents:
            for line in lines:
                if line.strip().startswith(agent.name):
                    allocations[agent] = line.strip()
        return allocations
