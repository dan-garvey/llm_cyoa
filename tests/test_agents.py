
import unittest
from unittest.mock import patch
from cyoa.agents import OverallAgent, CharacterAgent, ResponseAllocator

class TestAgents(unittest.TestCase):
    def setUp(self):
        self.mock_url = "http://localhost:9999"
        self.overall_agent = OverallAgent(server_url=self.mock_url)
        self.character_agents = [CharacterAgent(name, server_url=self.mock_url) for name in ['Alice', 'Bob', 'Eve']]
        self.allocator = ResponseAllocator()

    @patch("requests.post")
    def test_generate_response(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "Alice says something.\nBob says something.\nEve says something."}}]
        }
        prompt = "Test prompt."
        response = self.overall_agent.generate_response(prompt, self.character_agents)
        self.assertIn("Alice says something.", response)
        self.assertIn("Bob says something.", response)
        self.assertIn("Eve says something.", response)

    @patch("requests.post")
    def test_generate_dialogue(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "Dialogue for character."}}]
        }
        context = "Some context."
        for agent in self.character_agents:
            dialogue = agent.generate_dialogue(context)
            self.assertEqual(dialogue, "Dialogue for character.")

    def test_allocate(self):
        response = "Adventure: Test\nAlice says something.\nBob says something.\nEve says something."
        allocations = self.allocator.allocate(response, self.character_agents)
        self.assertEqual(len(allocations), 3)
        for agent in self.character_agents:
            self.assertIn(agent, allocations)
            self.assertIn(agent.name, allocations[agent])

if __name__ == "__main__":
    unittest.main()
