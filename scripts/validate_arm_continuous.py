#!/usr/bin/env python3
"""
机械臂连续运动验证脚本

三步验证流程:
  1. 快照前 → 记录所有 arm topic 状态
  2. 执行   → 通过 LangGraph API 发送 "机械臂向下移动一点点"
  3. 快照后 → 再次记录所有 arm topic 状态
  4. 对比   → 分析差异

用法:
  # 直接运行（启动服务 + 执行 + 清理）
  ./scripts/validate_arm_continuous.py

  # 手动分步（服务已运行）
  ./scripts/validate_arm_continuous.py snapshot
  ./scripts/validate_arm_continuous.py execute
  ./scripts/validate_arm_continuous.py compare
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime

os.environ["LD_LIBRARY_PATH"] = "/opt/robot/env/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
PYTHON = "/opt/robot/env/bin/python"
SNAPSHOT_FILE = "/tmp/arm_snapshot_before.json"
SNAPSHOT_AFTER_FILE = "/tmp/arm_snapshot_after.json"
RESULT_FILE = "/tmp/arm_execution_result.json"

ARM_TOPICS = [
    "/topic_arm_current_robot_eef_pose_0_283",
    "/topic_arm_current_robot_height_z_0_283",
    "/topic_arm_whole_body_current_joints_status_0_283",
    "/topic_arm_current_eef_pose_0_283",
    # 控制话题 — 看是否有 echo
    "/topic_arm_move_up_down_z_0_283",
    "/topic_arm_robot_action_0_283",
]


def source_ros2():
    """获取 ROS2 环境变量"""
    try:
        result = subprocess.run(
            ["bash", "-c", "source /opt/ros/jazzy/setup.bash && env | grep ROS_",
             "&& echo '---' && env | grep AMENT_"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout
    except Exception as e:
        return f"Error sourcing ROS2: {e}"


def snapshot_topics(output_file: str) -> dict:
    """订阅所有 arm topic 获取当前数据"""
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    rclpy.init()
    node = Node("arm_validator")
    results: dict[str, str] = {}
    subs = []

    def make_cb(topic: str):
        def cb(msg):
            if topic not in results:
                results[topic] = msg.data
        return cb

    for topic in ARM_TOPICS:
        try:
            sub = node.create_subscription(String, topic, make_cb(topic), 1)
            subs.append(sub)
        except Exception as e:
            results[topic] = f"<subscribe error: {e}>"

    # Wait for data
    for _ in range(200):
        rclpy.spin_once(node, timeout_sec=0.05)
        if len([k for k in results if not k.startswith("<")]) >= 3:
            break

    node.destroy_node()
    rclpy.shutdown()

    # Parse JSON when possible
    parsed = {}
    for topic, data in results.items():
        try:
            parsed[topic] = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            parsed[topic] = {"raw": data[:500] if data else "<no data>"}

    # Also add topics we didn't receive data from
    for topic in ARM_TOPICS:
        if topic not in parsed:
            parsed[topic] = {"_status": "no_data"}

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "ros2_env": source_ros2(),
        "topics": parsed,
    }
    with open(output_file, "w") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    return snapshot


def execute_command():
    """通过 LangGraph API 发送指令"""
    import requests

    # 启动 LangGraph 服务
    service_proc = subprocess.Popen(
        [
            "bash", "-c",
            f"cd /opt/seva && "
            f"source /opt/ros/jazzy/setup.bash && "
            f"LD_LIBRARY_PATH=/opt/robot/env/lib:$LD_LIBRARY_PATH "
            f"{PYTHON} run_agent.py --config config.yaml --port 8767"
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(3)

    try:
        # 健康检查
        for attempt in range(10):
            try:
                r = requests.get("http://127.0.0.1:8767/status", timeout=2)
                if r.status_code == 200:
                    break
            except:
                pass
            time.sleep(1)
        else:
            result = {"success": False, "message": "服务未能在10秒内启动"}
            with open(RESULT_FILE, "w") as f:
                json.dump(result, f)
            return result

        # 发送指令
        resp = requests.post(
            "http://127.0.0.1:8767/brain/task/text",
            json={"text": "机械臂向下移动一点点", "confirmed": True},
            timeout=30,
        )
        result = resp.json()
    except Exception as e:
        result = {"success": False, "message": f"请求失败: {e}"}
    finally:
        service_proc.terminate()
        service_proc.wait(timeout=5)

    with open(RESULT_FILE, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def compare_snapshots():
    """对比执行前后的 topic 数据"""
    try:
        with open(SNAPSHOT_FILE) as f:
            before = json.load(f)
    except FileNotFoundError:
        print("❌ 未找到快照前文件，请先运行 snapshot")
        return

    try:
        with open(SNAPSHOT_AFTER_FILE) as f:
            after = json.load(f)
    except FileNotFoundError:
        print("❌ 未找到快照后文件，请先运行 execute")
        return

    try:
        with open(RESULT_FILE) as f:
            exec_result = json.load(f)
    except FileNotFoundError:
        exec_result = {"message": "<未找到执行结果>"}

    print("=" * 70)
    print("  机械臂连续运动 — 执行前后对比报告")
    print("=" * 70)
    print(f"  快照前: {before['timestamp']}")
    print(f"  快照后: {after['timestamp']}")
    print(f"  执行结果: {json.dumps(exec_result, indent=2, ensure_ascii=False)}")
    print()

    topics_before = before["topics"]
    topics_after = after["topics"]

    # Compare each topic
    for topic in ARM_TOPICS:
        print(f"\n{'─' * 70}")
        print(f"  📡 {topic}")

        tb = topics_before.get(topic, {})
        ta = topics_after.get(topic, {})

        if "_status" in tb and "_status" in ta:
            print(f"  ⚪ 无数据")
            continue
        if "_status" in tb:
            print(f"  🔴 执行前无数据，执行后有")
            print(f"     {json.dumps(ta, indent=4, ensure_ascii=False)[:500]}")
            continue
        if "_status" in ta:
            print(f"  🟢 执行前有数据，执行后无")
            continue

        # Comparison logic per topic type
        if "current_robot_eef_pose" in topic:
            _compare_eef_pose(tb, ta, "left_eef_pose")
            _compare_eef_pose(tb, ta, "right_eef_pose")
        elif "height_z" in topic:
            h_before = tb.get("height_z", "N/A")
            h_after = ta.get("height_z", "N/A")
            print(f"    高度 Z: {h_before} → {h_after}")
        elif "whole_body_current_joints" in topic:
            _compare_joints(tb, ta)
        else:
            # Generic comparison
            b_str = json.dumps(tb, ensure_ascii=False)[:200]
            a_str = json.dumps(ta, ensure_ascii=False)[:200]
            if b_str != a_str:
                print(f"  ✅ 数据发生变化")
                print(f"     前: {b_str}")
                print(f"     后: {a_str}")
            else:
                print(f"  ⚪ 数据未变化")

    print()
    print("=" * 70)
    print("  对比完成")
    print("=" * 70)


def _compare_eef_pose(before: dict, after: dict, key: str):
    """对比 EEF pose 中的位姿数据"""
    b_pose = before.get(key)
    a_pose = after.get(key)
    if not b_pose or not a_pose:
        print(f"  ⚪ {key}: 无数据")
        return

    b_pos = b_pose.get("position", [])
    a_pos = a_pose.get("position", [])

    if b_pos and a_pos and len(b_pos) == 3 and len(a_pos) == 3:
        deltas = [round(a_pos[i] - b_pos[i], 6) for i in range(3)]
        changed = any(abs(d) > 0.001 for d in deltas)
        status = "✅ 有变化" if changed else "⚪ 未变化"
        print(f"  {status} {key}.position:")
        print(f"     前: [{b_pos[0]:.4f}, {b_pos[1]:.4f}, {b_pos[2]:.4f}]")
        print(f"     后: [{a_pos[0]:.4f}, {a_pos[1]:.4f}, {a_pos[2]:.4f}]")
        print(f"     Δ:  [{deltas[0]:.4f}, {deltas[1]:.4f}, {deltas[2]:.4f}]")
        if changed:
            print(f"     预期 Δ: 向下 0.03m → [0, 0, -0.03]")
            print(f"     匹配预期: {'✅ 是' if abs(deltas[2] + 0.03) < 0.01 else '❌ 否'}")
    else:
        print(f"  ⚪ {key}.position: 格式不可比")


def _compare_joints(before: dict, after: dict):
    """对比关节数据"""
    # Generic comparison
    b_str = json.dumps(before, ensure_ascii=False)[:300]
    a_str = json.dumps(after, ensure_ascii=False)[:300]
    if b_str != a_str:
        print(f"  ✅ 关节数据有变化")
    else:
        print(f"  ⚪ 关节数据未变化")


def print_usage():
    print(f"用法: {sys.argv[0]} {{snapshot|execute|compare|all}}")
    print()
    print("  snapshot   — 快照执行前的 topic 状态")
    print("  execute    — 启动服务并执行 '机械臂向下移动一点点'")
    print("  compare    — 对比执行前后的数据")
    print("  all        — 全自动三步验证")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"

    if cmd == "snapshot":
        print("📸 快照执行前状态...")
        snap = snapshot_topics(SNAPSHOT_FILE)
        n = len([k for k, v in snap["topics"].items() if "_status" not in v])
        print(f"   已记录 {n}/{len(ARM_TOPICS)} 个话题")

    elif cmd == "execute":
        print("🚀 执行 LangGraph 指令...")
        result = execute_command()
        print(f"   结果: {json.dumps(result, indent=2, ensure_ascii=False)[:500]}")

        print("📸 快照执行后状态...")
        snap = snapshot_topics(SNAPSHOT_AFTER_FILE)
        n = len([k for k, v in snap["topics"].items() if "_status" not in v])
        print(f"   已记录 {n}/{len(ARM_TOPICS)} 个话题")

    elif cmd == "compare":
        compare_snapshots()

    elif cmd == "all":
        print("=" * 70)
        print("  步骤 1/3: 快照执行前状态")
        print("=" * 70)
        snapshot_topics(SNAPSHOT_FILE)
        print("  ✅ 完成\n")

        print("=" * 70)
        print("  步骤 2/3: 执行指令 + 快照执行后")
        print("=" * 70)
        result = execute_command()
        print(f"  执行结果: {json.dumps(result, indent=2, ensure_ascii=False)[:500]}")
        snapshot_topics(SNAPSHOT_AFTER_FILE)
        print("  ✅ 完成\n")

        print("=" * 70)
        print("  步骤 3/3: 对比分析")
        print("=" * 70)
        compare_snapshots()

    else:
        print_usage()
