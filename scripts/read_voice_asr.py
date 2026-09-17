#!/usr/bin/env python3
"""Read the text already recognized by autolife_robot_vision without opening the microphone."""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

TOPIC = os.environ.get("ZENG_ASR_TOPIC", "/text_voice_command")


class VoiceTextReader(Node):
    def __init__(self):
        super().__init__("zeng_agent_voice_text_reader")
        self.subscription = self.create_subscription(String, TOPIC, self.on_text, 10)

    def on_text(self, message):
        text = message.data.strip()
        if text:
            print(json.dumps({"text": text, "source": "autolife_robot_vision", "topic": TOPIC}, ensure_ascii=False), flush=True)


def main():
    rclpy.init()
    node = VoiceTextReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
