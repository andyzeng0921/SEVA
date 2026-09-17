import unittest
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from zeng_agent.nlu import IntentParser


class IntentParserTests(unittest.TestCase):
    def test_parse_navigate_to_waypoint(self):
        parsed = IntentParser().parse("去 home")

        self.assertEqual(parsed.intent, "navigate_to_waypoint")
        self.assertEqual(parsed.slots["name"], "home")
        self.assertFalse(parsed.requires_confirmation)

    def test_parse_save_waypoint(self):
        parsed = IntentParser().parse("保存点位 kitchen")

        self.assertEqual(parsed.intent, "save_waypoint")
        self.assertEqual(parsed.slots["name"], "kitchen")

    def test_parse_arm_reset_preset(self):
        parsed = IntentParser().parse("机械臂复位")

        self.assertEqual(parsed.intent, "arm_preset")
        self.assertEqual(parsed.slots["action_name"], "reset")

    def test_parse_unsupported_freeform_arm_command_requires_confirmation(self):
        parsed = IntentParser().parse("机械臂画一个圆")

        self.assertEqual(parsed.intent, "unsupported")
        self.assertTrue(parsed.requires_confirmation)


if __name__ == "__main__":
    unittest.main()
