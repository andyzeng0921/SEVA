#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

def build_reset_topic(topic_node_id: str) -> str:
    return f"control_reset_{topic_node_id}"


def build_action_topic(topic_node_id: str) -> str:
    return f"topic_arm_robot_action_{topic_node_id}"


def build_action_message(action_name: str) -> dict[str, str]:
    return {"action_type": "play", "action_name": action_name}


class ArmCommandNode(Node):
    def __init__(self, topic_node_id: str) -> None:
        super().__init__("zeng_agent_arm_command")
        self._reset_pub = self.create_publisher(String, build_reset_topic(topic_node_id), 10)
        self._action_pub = self.create_publisher(String, build_action_topic(topic_node_id), 10)

    def publish_reset(self) -> None:
        self._reset_pub.publish(String())

    def publish_action(self, action_name: str) -> None:
        self._action_pub.publish(String(data=json.dumps(build_action_message(action_name), ensure_ascii=False)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish live arm reset/preset commands")
    parser.add_argument("mode", choices=["reset", "play"])
    parser.add_argument("--topic-node-id", required=True)
    parser.add_argument("--action-name", default=None)
    args = parser.parse_args()

    rclpy.init()
    node = ArmCommandNode(topic_node_id=args.topic_node_id)
    try:
        publisher = node._reset_pub if args.mode == "reset" else node._action_pub
        deadline = time.monotonic() + 5.0
        while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
        if publisher.get_subscription_count() == 0:
            print(json.dumps({"success": False, "message": "No arm-control subscriber"}))
            return 2

        if args.mode == "reset":
            node.publish_reset()
        else:
            if not args.action_name:
                raise SystemExit("--action-name is required for play mode")
            node.publish_action(args.action_name)
        rclpy.spin_once(node, timeout_sec=0.5)
        print(json.dumps({"success": True, "message": f"Published {args.mode}"}))
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
