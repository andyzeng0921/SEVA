#!/usr/bin/env python3
"""检查 ROS2 arm 话题的 JSON 消息格式"""
import json
import sys
import os

os.environ['LD_LIBRARY_PATH'] = '/opt/robot/env/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

TOPICS = [
    '/topic_arm_current_robot_eef_pose_0_283',
    '/topic_arm_move_up_down_z_0_283',
    '/topic_arm_move_eef_pose_in_robot_frame_0_283',
    '/topic_arm_whole_body_current_joints_status_0_283',
    '/topic_arm_whole_body_target_joints_position_0_283',
    '/topic_arm_current_robot_height_z_0_283',
    '/topic_arm_robot_action_0_283',
    '/topic_arm_move_joints_trajectory_0_283',
]

rclpy.init()
node = Node('topic_scanner')
results = {}

def make_cb(topic_name):
    def cb(msg):
        if topic_name not in results:
            try:
                results[topic_name] = json.loads(msg.data)
            except json.JSONDecodeError:
                results[topic_name] = {"raw_string": msg.data[:500]}
    return cb

subs = []
for topic in TOPICS:
    sub = node.create_subscription(String, topic, make_cb(topic), 1)
    subs.append(sub)

print("扫描话题中...")
for _ in range(200):
    rclpy.spin_once(node, timeout_sec=0.05)
    if len(results) >= 4:
        break

node.destroy_node()
rclpy.shutdown()

if not results:
    print("未收到任何话题消息")
    sys.exit(1)

for topic, data in sorted(results.items()):
    print(f"\n{'='*60}")
    print(f"📡 {topic}")
    print(f"{'='*60}")
    print(json.dumps(data, indent=2, ensure_ascii=False)[:800])
