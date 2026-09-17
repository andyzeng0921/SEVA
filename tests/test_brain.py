import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from zeng_agent.brain import BrainOrchestrator
from zeng_agent.config import AgentConfig
from zeng_agent.executor import AgentExecutor
from zeng_agent.nlu import IntentParser


class BrainOrchestratorTests(unittest.TestCase):
    def build_brain(self, config=None):
        config = config or AgentConfig(waypoint_names=["home", "dock"])
        executor = AgentExecutor(config=config, waypoint_names=config.waypoint_names)
        return BrainOrchestrator(config=config, parser=IntentParser(), executor=executor)

    def test_status_exposes_framework_and_skills(self):
        status = self.build_brain().status()

        self.assertTrue(status["enabled"])
        self.assertEqual(status["framework"], "langgraph_ready")
        self.assertIn("navigate_to_waypoint", {skill["intent"] for skill in status["skills"]})

    def test_motion_task_requires_confirmation(self):
        result = self.build_brain().run_text_task("\u53bb home")

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["steps"][0]["status"], "blocked")
        self.assertIn("confirmation", result["message"])

    def test_confirmed_motion_task_executes(self):
        result = self.build_brain().run_text_task("\u53bb home", confirmed=True)

        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["steps"][0]["result"]["success"])

    def test_long_sequence_executes_in_order(self):
        result = self.build_brain().run_text_task("\u67e5\u8be2\u72b6\u6001 \u7136\u540e \u67e5\u8be2\u70b9\u4f4d", confirmed=False)

        self.assertEqual(result["status"], "completed")
        self.assertEqual([step["intent"] for step in result["steps"]], ["robot_status", "list_waypoints"])

    def test_stop_is_safety_override_without_confirmation(self):
        result = self.build_brain().run_text_task("\u505c\u6b62")

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["steps"][0]["intent"], "stop_navigation")

    def test_max_steps_blocks_oversized_task(self):
        config = AgentConfig()
        config.brain.max_steps = 1
        result = self.build_brain(config).run_text_task("\u67e5\u8be2\u72b6\u6001 \u7136\u540e \u67e5\u8be2\u70b9\u4f4d")

        self.assertEqual(result["status"], "blocked")
        self.assertIn("max_steps", result["message"])


if __name__ == "__main__":
    unittest.main()
