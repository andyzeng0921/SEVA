import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from zeng_agent.live_arm import build_action_message, build_action_topic, build_reset_topic


class LiveArmTests(unittest.TestCase):
    def test_build_reset_topic(self):
        self.assertEqual(build_reset_topic("0_283"), "control_reset_0_283")

    def test_build_action_topic(self):
        self.assertEqual(build_action_topic("0_283"), "topic_arm_robot_action_0_283")

    def test_build_action_message(self):
        payload = build_action_message("wave")

        self.assertEqual(payload["action_type"], "play")
        self.assertEqual(payload["action_name"], "wave")


if __name__ == "__main__":
    unittest.main()
