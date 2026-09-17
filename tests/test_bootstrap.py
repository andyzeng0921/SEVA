import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from zeng_agent.bootstrap import build_service
from zeng_agent.backends import CompositeControlBackend
from zeng_agent.config import AgentConfig
from zeng_agent.executor import DryRunBackend


class BootstrapTests(unittest.TestCase):
    def test_build_service_uses_dry_run_backend_by_default(self):
        service = build_service(AgentConfig(), waypoint_names=["home"])

        self.assertIsInstance(service.executor.backend, DryRunBackend)

    def test_build_service_uses_composite_backend_in_live_mode(self):
        config = AgentConfig(dry_run=False, arm_preset_commands={"reset": "echo reset", "stop": "echo stop"})
        service = build_service(config, waypoint_names=["home"], runner=lambda command: (0, "ok", ""))

        self.assertIsInstance(service.executor.backend, CompositeControlBackend)

    def test_build_service_wires_local_asr_provider_when_transcriber_command_exists(self):
        config = AgentConfig()
        config.local_asr.transcriber_command = "echo hello {wav_path}"

        service = build_service(config, waypoint_names=["home"])

        self.assertIsNotNone(service.local_asr_provider)

    def test_build_service_populates_arm_commands_from_topic_node_id(self):
        config = AgentConfig(dry_run=False, topic_node_id="0_283")

        service = build_service(config, waypoint_names=["home"], runner=lambda command: (0, "ok", ""))

        self.assertIn("reset", service.config.arm_preset_commands)
        self.assertIn("wave", service.config.arm_preset_commands)
        self.assertIn("--topic-node-id 0_283", service.config.arm_preset_commands["wave"])


if __name__ == "__main__":
    unittest.main()
