from __future__ import annotations


def build_reset_topic(topic_node_id: str) -> str:
    return f"control_reset_{topic_node_id}"


def build_action_topic(topic_node_id: str) -> str:
    return f"topic_arm_robot_action_{topic_node_id}"


def build_action_message(action_name: str) -> dict[str, str]:
    return {
        "action_type": "play",
        "action_name": action_name,
    }
