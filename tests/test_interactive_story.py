import unittest
from cyoa.agent_orchestrator import AgentOrchestrator

class TestInteractiveStoryLoop(unittest.TestCase):
    def setUp(self):
        self.model_path = "meta-llama/Llama-3.2-3B-Instruct"
        self.orchestrator = AgentOrchestrator(
            self.model_path,
            storyteller_gpu=0,
            director_gpu=1,
            character_gpu=2
        )
        self.orchestrator.start_storyteller_and_director()

    def tearDown(self):
        self.orchestrator.stop_all()

    def test_interactive_story_loop(self):
        user_name = "Astra Vey"
        user_background = "A mysterious wanderer with a silver compass, seeking the lost city of mirrors."
        user_inputs = [
            "Astra Vey looks around the forest, searching for clues.",
            "She asks Kael Darkhaven about the city of mirrors.",
            "Astra Vey decides to trust Kael and follow him deeper into the woods."
        ]
        story = self.orchestrator.interactive_story_loop(
            user_name,
            user_background,
            user_inputs,
            max_turns=len(user_inputs)
        )
        self.assertIn(user_name, story)
        print("\nFinal Story Output:\n", story)

if __name__ == "__main__":
    unittest.main()
