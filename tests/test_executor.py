import unittest
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from zeng_agent.config import AgentConfig
from zeng_agent.executor import AgentExecutor
from zeng_agent.models import ExecutionRequest


class AgentExecutorTests(unittest.TestCase):
    def test_dry_run_navigate_known_waypoint(self):
        executor = AgentExecutor(
            config=AgentConfig(),
            waypoint_names=["home", "dock"],
        )

        result = executor.execute(
            ExecutionRequest(intent="navigate_to_waypoint", validated_slots={"name": "home"}, mode="dry_run")
        )

        self.assertTrue(result.success)
        self.assertEqual(result.data["mode"], "dry_run")
        self.assertEqual(result.data["action"], "navigate_to_waypoint")

    def test_reject_unknown_waypoint(self):
        executor = AgentExecutor(
            config=AgentConfig(),
            waypoint_names=["dock"],
        )

        result = executor.execute(
            ExecutionRequest(intent="navigate_to_waypoint", validated_slots={"name": "home"}, mode="dry_run")
        )

        self.assertFalse(result.success)
        self.assertIn("unknown waypoint", result.message.lower())

    def test_dry_run_arm_preset(self):
        executor = AgentExecutor(
            config=AgentConfig(),
            waypoint_names=["home"],
        )

        result = executor.execute(
            ExecutionRequest(intent="arm_preset", validated_slots={"action_name": "reset"}, mode="dry_run")
        )

        self.assertTrue(result.success)
        self.assertEqual(result.data["action"], "arm_preset")
        self.assertEqual(result.data["preset"], "reset")

    def test_dry_run_without_catalog_allows_navigation_name(self):
        executor = AgentExecutor(
            config=AgentConfig(),
            waypoint_names=[],
        )

        result = executor.execute(
            ExecutionRequest(intent="navigate_to_waypoint", validated_slots={"name": "temporary_home"}, mode="dry_run")
        )

        self.assertTrue(result.success)
        self.assertEqual(result.data["name"], "temporary_home")


if __name__ == "__main__":
    unittest.main()
