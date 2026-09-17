#!/usr/bin/env python3
"""
publish_arm_continuous.py — 连续运动控制发布脚本 (含安全校验)

模式:
    # 机器人身体升降
    ./scripts/publish_arm_continuous.py --direction down --amount 0.05

    # 手臂 EEF 位移
    ./scripts/publish_arm_continuous.py --direction left --amount 0.05 --limb left

    # 一手上下一手交叉 (身体前方安全)
    ./scripts/publish_arm_continuous.py --mode cross_wave --phase left_up
    ./scripts/publish_arm_continuous.py --mode cross_wave --phase right_up

    # 跳过安全校验 (危险!)
    ./scripts/publish_arm_continuous.py --direction down --amount 0.05 --no-safety

安全说明:
    - up/down: 控制身体高度，范围 -0.56~0.0
    - left/right/forward/backward: 控制手臂 EEF，包含空间位置安全检查
    - cross_wave: 使用经标定的安全关节角度，前方交叉挥手
"""
import argparse
import json
import os
import sys
import time

os.environ["LD_LIBRARY_PATH"] = "/opt/robot/env/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# 安全模块路径
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from zeng_agent.arm_safety import (
    check_power_on,
    validate_joint_angles,
    validate_eef_position,
    make_cross_wave_joints,
    make_whole_body_joint_msg,
    LEFT_ARM_FRONT_UP,
    RIGHT_ARM_DOWN,
    LEFT_ARM_DOWN,
    RIGHT_ARM_FRONT_UP,
)


DIRECTION_VECTORS = {
    "up":      [0.0,  0.0,  1.0],
    "down":    [0.0,  0.0, -1.0],
    "left":    [0.0,  1.0,  0.0],
    "right":   [0.0, -1.0,  0.0],
    "forward": [1.0,  0.0,  0.0],
    "backward":[-1.0, 0.0,  0.0],
}

LIMB_POSE_KEYS = {
    "left":  "left_eef_pose",
    "right": "right_eef_pose",
}


def get_current_pose(node: Node, topic: str, timeout: float = 3.0) -> dict | None:
    result = []
    def cb(msg):
        try:
            result.append(json.loads(msg.data))
        except json.JSONDecodeError:
            pass
    sub = node.create_subscription(String, topic, cb, 1)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not result:
        rclpy.spin_once(node, timeout_sec=0.05)
    if sub:
        node.destroy_subscription(sub)
    return result[0] if result else None


def run_safety_checks(node, nid, limb, target_pose=None):
    """执行全部安全校验，返回 (ok, message)"""
    # 1) 上电检查
    power = check_power_on(node, nid)
    if not power.allowed:
        return False, power.reason

    # 2) EEF 位置检查 (如果有目标位置)
    if target_pose:
        eef_topic = f"/topic_arm_current_robot_eef_pose_{nid}"
        pose_data = get_current_pose(node, eef_topic, timeout=3.0)
        if pose_data:
            lp = pose_data["left_eef_pose"]["position"]
            rp = pose_data["right_eef_pose"]["position"]
            result = validate_eef_position(lp, rp)
            if not result.allowed:
                return False, result.reason

    return True, "安全检查通过 ✅"


def do_cross_wave(node, nid, phase, safety=True):
    """执行一手上下一手交叉

    phase="left_up":  左手上 + 右手下
    phase="right_up": 左手下 + 右手上
    """
    # 安全检查
    if safety:
        ok, msg = run_safety_checks(node, nid, "both")
        if not ok:
            return {"success": False, "message": f"安全检查未通过: {msg}", "data": {}}

    left_arm, right_arm = make_cross_wave_joints(phase)

    # 关节角度校验
    if safety:
        joint_check = validate_joint_angles(left_arm, right_arm)
        if not joint_check.allowed:
            return {
                "success": False,
                "message": f"关节角度不安全: {joint_check.reason}",
                "warnings": joint_check.warnings,
                "data": {},
            }

    topic = f"/topic_arm_whole_body_target_joints_position_{nid}"
    msg_str = make_whole_body_joint_msg(left_arm, right_arm)

    pub = node.create_publisher(String, topic, 1)
    time.sleep(0.1)
    msg = String()
    msg.data = msg_str
    pub.publish(msg)
    time.sleep(0.5)
    node.destroy_publisher(pub)

    phase_label = "左臂上↑ + 右臂下↓" if phase == "left_up" else "左臂下↓ + 右臂上↑"
    return {
        "success": True,
        "message": f"前方交叉挥手: {phase_label}",
        "data": {
            "action": "cross_wave",
            "phase": phase,
            "left_arm_target": left_arm,
            "right_arm_target": right_arm,
            "warnings": joint_check.warnings if safety else [],
            "safety": "enabled" if safety else "DISABLED",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="机器人运动控制 (含安全校验)")
    # 新增
    parser.add_argument("--mode", default="continuous", choices=["continuous", "cross_wave", "reset"],
                        help="运动模式: continuous=方向位移, cross_wave=交叉挥手, reset=复位")
    parser.add_argument("--phase", choices=["left_up", "right_up"],
                        help="cross_wave 模式下的动作相位")
    # 原有
    parser.add_argument("--direction", choices=list(DIRECTION_VECTORS.keys()),
                        help="移动方向 (continuous 模式下必填)")
    parser.add_argument("--amount", type=float, default=0.05,
                        help="移动距离（米）")
    parser.add_argument("--limb", default="left", choices=["left", "right"],
                        help="手臂侧")
    parser.add_argument("--topic-node-id", default="0_283")
    parser.add_argument("--speed", type=float, default=0.1)
    parser.add_argument("--pub-timeout", type=float, default=1.0)
    parser.add_argument("--no-safety", action="store_true",
                        help="跳过安全校验 (危险!)")
    args = parser.parse_args()

    rclpy.init()
    node = Node("arm_control")
    safety = not args.no_safety

    try:
        # ── 模式: cross_wave ────────────────────────────
        if args.mode == "cross_wave":
            if not args.phase:
                output = {"success": False, "message": "cross_wave 模式需要 --phase 参数 (left_up 或 right_up)", "data": {}}
                print(json.dumps(output, ensure_ascii=False))
                sys.exit(1)
            output = do_cross_wave(node, args.topic_node_id, args.phase, safety=safety)
            print(json.dumps(output, ensure_ascii=False))
            return

        # ── 模式: reset ────────────────────────────────
        if args.mode == "reset":
            for topic, data in [
                (f"/control_reset_{args.topic_node_id}", ""),
                (f"/topic_arm_robot_action_{args.topic_node_id}",
                 json.dumps({"action_type": "play", "action_name": "reset"})),
                (f"/topic_arm_move_up_down_z_{args.topic_node_id}",
                 json.dumps({"height_z": 0.0, "move_joint_waist_pitch": True})),
            ]:
                pub = node.create_publisher(String, topic, 1)
                time.sleep(0.1)
                pub.publish(String(data=data))
                node.destroy_publisher(pub)
            output = {"success": True, "message": "复位指令已发送 (control_reset + preset reset + height_z=0)", "data": {}}
            print(json.dumps(output, ensure_ascii=False))
            return

        # ── 模式: continuous ────────────────────────────
        if not args.direction:
            output = {"success": False, "message": "continuous 模式需要 --direction 参数", "data": {}}
            print(json.dumps(output, ensure_ascii=False))
            sys.exit(1)

        # 安全检查
        if safety:
            ok, msg = run_safety_checks(node, args.topic_node_id, args.limb)
            if not ok:
                output = {"success": False, "message": f"安全检查未通过: {msg}", "data": {}}
                print(json.dumps(output, ensure_ascii=False))
                sys.exit(1)

        # ── 分支 A: up/down → 控制机器人整体高度 ──
        if args.direction in ("up", "down"):
            height_fb_topic = f"/topic_arm_current_robot_height_z_{args.topic_node_id}"
            height_data = get_current_pose(node, height_fb_topic, timeout=3.0)
            if height_data is None or "height_z" not in height_data:
                output = {"success": False, "message": "无法获取当前 height_z", "data": {}}
                print(json.dumps(output))
                sys.exit(1)

            current_height_z = height_data["height_z"]
            sign = -1.0 if args.direction == "down" else 1.0
            target_height_z = max(-0.56, min(0.0, current_height_z + sign * args.amount))

            control_topic = f"/topic_arm_move_up_down_z_{args.topic_node_id}"
            control_msg = json.dumps({"height_z": target_height_z, "move_joint_waist_pitch": True})

            pub = node.create_publisher(String, control_topic, 1)
            time.sleep(0.1)
            pub.publish(String(data=control_msg))
            time.sleep(args.pub_timeout)
            node.destroy_publisher(pub)

            output = {
                "success": True,
                "message": f"身体高度控制: {args.direction} {args.amount}m (height_z={target_height_z:.2f})",
                "data": {"action": "robot_height", "direction": args.direction,
                         "amount_meters": args.amount, "current_height_z": current_height_z,
                         "target_height_z": target_height_z, "control_topic": control_topic},
            }

        # ── 分支 B: left/right/forward/backward → EEF 位移 ──
        else:
            current_pose_topic = f"/topic_arm_current_robot_eef_pose_{args.topic_node_id}"
            pose_data = get_current_pose(node, current_pose_topic, timeout=3.0)
            if not pose_data:
                output = {"success": False, "message": f"无法获取 EEF Pose (topic: {current_pose_topic})", "data": {}}
                print(json.dumps(output))
                sys.exit(1)

            limb_key = LIMB_POSE_KEYS[args.limb]
            cur_pos = pose_data[limb_key]["position"]
            cur_rot = pose_data[limb_key]["rotation"]

            vec = DIRECTION_VECTORS[args.direction]
            displacement = [v * args.amount for v in vec]
            target_pos = [cur_pos[i] + displacement[i] for i in range(3)]

            # 安全检查: 预测目标位置是否越界
            if safety:
                if args.limb == "left":
                    eef_check = validate_eef_position(tuple(target_pos), tuple(pose_data["right_eef_pose"]["position"]))
                else:
                    eef_check = validate_eef_position(tuple(pose_data["left_eef_pose"]["position"]), tuple(target_pos))
                if not eef_check.allowed:
                    output = {"success": False, "message": f"目标位置危险: {eef_check.reason}", "data": {}}
                    print(json.dumps(output, ensure_ascii=False))
                    sys.exit(1)

            control_topic = f"/topic_arm_move_eef_pose_in_robot_frame_{args.topic_node_id}"
            control_msg = json.dumps({
                "position": target_pos, "rotation": cur_rot,
                "speed": args.speed, "limb": args.limb,
            })

            pub = node.create_publisher(String, control_topic, 1)
            time.sleep(0.1)
            pub.publish(String(data=control_msg))
            time.sleep(args.pub_timeout)
            node.destroy_publisher(pub)

            output = {
                "success": True,
                "message": f"手臂 EEF 位移: {args.limb}臂 {args.direction} {args.amount}m",
                "data": {"action": "arm_eef_move", "limb": args.limb, "direction": args.direction,
                         "amount_meters": args.amount, "current_position": cur_pos,
                         "target_position": target_pos, "displacement": displacement,
                         "control_topic": control_topic},
            }

        print(json.dumps(output, ensure_ascii=False))

    except Exception as e:
        output = {"success": False, "message": f"执行失败: {e}", "data": {}}
        print(json.dumps(output))
        sys.exit(1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
