

from vllm import LLM, SamplingParams

class OverallAgent:
    def __init__(self, model_path="meta-llama/Meta-Llama-3-8B-Instruct"):
        self.llm = LLM(model=model_path)
        self.sampling_params = SamplingParams(max_tokens=256)

    def generate_response(self, prompt, character_agents):
        char_names = ', '.join([agent.name for agent in character_agents])
        full_prompt = f"Adventure: {prompt}\nCharacters: {char_names}\nRespond with dialogue for each character."
        outputs = self.llm.generate([full_prompt], self.sampling_params)
        return outputs[0].outputs[0].text


class CharacterAgent:
    def __init__(self, name, model_path="meta-llama/Meta-Llama-3-8B-Instruct"):
        self.name = name
        self.llm = LLM(model=model_path)
        self.sampling_params = SamplingParams(max_tokens=64)

    def generate_dialogue(self, context):
        prompt = f"{self.name}: {context}"
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text



class ResponseAllocator:
    def allocate(self, overall_response, character_agents):
        lines = overall_response.split('\n')
        allocations = {}
        for agent in character_agents:
            for line in lines:
                if line.strip().startswith(agent.name):
                    allocations[agent] = line.strip()
        return allocations
