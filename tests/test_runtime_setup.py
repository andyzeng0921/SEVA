import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from zeng_agent.runtime_setup import build_arm_preset_commands, build_whispercpp_transcriber_command


class RuntimeSetupTests(unittest.TestCase):
    def test_build_whispercpp_transcriber_command_contains_expected_paths(self):
        command = build_whispercpp_transcriber_command(
            python_bin="/env/python",
            helper_script="/agent/scripts/run_whisper_cpp.py",
            whisper_bin="/agent/vendor/whisper.cpp/build/bin/whisper-cli",
            model_path="/agent/models/ggml-base.bin",
            language="zh",
        )

        self.assertIn("/env/python", command)
        self.assertIn("/agent/scripts/run_whisper_cpp.py", command)
        self.assertIn("--binary /agent/vendor/whisper.cpp/build/bin/whisper-cli", command)
        self.assertIn("--model /agent/models/ggml-base.bin", command)
        self.assertIn("--language zh", command)
        self.assertIn("{wav_path}", command)

    def test_build_arm_preset_commands_for_robot_topic_node_id(self):
        commands = build_arm_preset_commands(
            python_bin="/env/python",
            helper_script="/agent/scripts/publish_arm_command.py",
            topic_node_id="0_283",
            presets=["wave", "zero_position"],
        )

        self.assertIn("reset", commands)
        self.assertIn("stop", commands)
        self.assertIn("wave", commands)
        self.assertIn("--topic-node-id 0_283", commands["wave"])
        self.assertIn("--action-name wave", commands["wave"])


if __name__ == "__main__":
    unittest.main()
