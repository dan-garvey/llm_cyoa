

from cyoa.agents import OverallAgent, CharacterAgent, ResponseAllocator
import os
import requests
from scripts.spawn_and_connect import spawn_vllm_servers

# Hardcoded Hugging Face token (replace with your token)
HF_TOKEN = "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"

def main():

    # Use Hugging Face repo path directly for vllm
    model_path = "NousResearch/Meta-Llama-3-8B-Instruct"

    # Spawn vllm servers for overall agent and each character
    names = ['Alice', 'Bob', 'Eve']
    num_servers = 1 + len(names)
    managers, urls = spawn_vllm_servers(model_path, num_servers)
    overall_agent = OverallAgent(server_url=urls[0])
    character_agents = [CharacterAgent(name, server_url=urls[i+1]) for i, name in enumerate(names)]
    allocator = ResponseAllocator()

    # Example adventure prompt
    prompt = "You enter a mysterious cave. What do you do?"
    overall_response = overall_agent.generate_response(prompt, character_agents)
    allocations = allocator.allocate(overall_response, character_agents)

    print("\n--- Adventure Response ---")
    print(overall_response)
    print("\n--- Character Dialogues ---")
    for agent in character_agents:
        context = allocations.get(agent, prompt)
        dialogue = agent.generate_dialogue(context)
        print(f"{agent.name} reads: {dialogue}")

    # Stop all vllm servers
    for m in managers:
        m.stop()

if __name__ == "__main__":
    main()
