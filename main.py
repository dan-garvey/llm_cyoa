
from cyoa.agents import OverallAgent, CharacterAgent, ResponseAllocator
import os


import requests

# Hardcoded Hugging Face token (replace with your token)
HF_TOKEN = "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"

def download_model(model_path):
    # Download the most recent Llama-3 8B GGUF from Hugging Face
    if os.path.exists(model_path):
        return
    url = "https://huggingface.co/TheBloke/Llama-3-8B-GGUF/resolve/main/llama-3-8B.Q4_K_M.gguf"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    print(f"Downloading Llama-3 8B model from Hugging Face to {model_path}...")
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Model download complete.")
    else:
        print(f"Failed to download model: {response.status_code}")
        exit(1)


def main():
    model_path = "models/llama-3-8B.Q4_K_M.gguf"
    download_model(model_path)

    # Initialize agents with Llama-3 8B model
    overall_agent = OverallAgent(model_path=model_path)
    character_agents = [CharacterAgent(name, model_path=model_path) for name in ['Alice', 'Bob', 'Eve']]
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

if __name__ == "__main__":
    main()
