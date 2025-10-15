import unittest
from cyoa.agents import OverallAgent, CharacterAgent, ResponseAllocator

class TestAgents(unittest.TestCase):
    def setUp(self):
        self.overall_agent = OverallAgent()
        self.character_agents = [CharacterAgent(name) for name in ['Alice', 'Bob', 'Eve']]
        self.allocator = ResponseAllocator()

    def test_generate_response(self):
        prompt = "Test prompt."
        response = self.overall_agent.generate_response(prompt, self.character_agents)
        self.assertIn("Alice says something.", response)
        self.assertIn("Bob says something.", response)
        self.assertIn("Eve says something.", response)

    def test_allocate(self):
        response = "Adventure: Test\nAlice says something.\nBob says something.\nEve says something."
        allocations = self.allocator.allocate(response, self.character_agents)
        self.assertEqual(len(allocations), 3)
        for agent in self.character_agents:
            self.assertIn(agent, allocations)
            self.assertIn(agent.name, allocations[agent])

if __name__ == "__main__":
    unittest.main()
