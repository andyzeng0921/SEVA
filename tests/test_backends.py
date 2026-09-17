import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from zeng_agent.backends import ArmCommandBackend, Ros2ShellBackend
from zeng_agent.config import AgentConfig


class FakeRunner:
    def __init__(self) -> None:
        self.commands = []

    def run(self, command: str):
        self.commands.append(command)
        return 0, "success: true\nmessage: ok\nnames:\n- home\n- dock\n", ""


class BackendTests(unittest.TestCase):
    def test_ros2_navigate_formats_service_command(self):
        runner = FakeRunner()
        backend = Ros2ShellBackend(config=AgentConfig(), runner=runner.run)

        backend.navigate_to_waypoint("home")

        self.assertTrue(runner.commands)
        self.assertIn("ros2 service call /navigate_to_waypoint", runner.commands[-1])
        self.assertIn("{name: home}", runner.commands[-1])

    def test_arm_backend_uses_configured_preset_command(self):
        runner = FakeRunner()
        config = AgentConfig(arm_preset_commands={"reset": "echo arm-reset"})
        backend = ArmCommandBackend(config=config, runner=runner.run)

        result = backend.arm_preset("reset")

        self.assertTrue(result.success)
        self.assertEqual(runner.commands[-1], "echo arm-reset")

    def test_arm_backend_uses_configured_stop_command(self):
        runner = FakeRunner()
        config = AgentConfig(arm_preset_commands={"stop": "echo arm-stop"})
        backend = ArmCommandBackend(config=config, runner=runner.run)

        result = backend.arm_stop()

        self.assertTrue(result.success)
        self.assertEqual(runner.commands[-1], "echo arm-stop")


if __name__ == "__main__":
    unittest.main()
