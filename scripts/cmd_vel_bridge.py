#!/usr/bin/env python3
"""
cmd_vel_bridge.py — 底盘速度控制桥接

作为 ROS2 节点运行，接收来自 agent 的 cmd_vel 指令，
通过 GVController.publish_velocity_command() 发布到 /topic_gv_target_cmd_vel_0_283。

由于 CycloneDDS loopback 限制，这是唯一的 ROS2 节点入口。
与 ChassisController 通过 STDIN JSON pipe 通信。
"""
import sys, os, json, time, signal, threading

os.environ["LD_LIBRARY_PATH"] = "/opt/robot/env/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

import rclpy
from rclpy.executors import MultiThreadedExecutor

# GVController 在 autolife_robot_arm examples 子包中
_examples_path = "/opt/robot/env/lib/python3.12/site-packages/autolife_robot_arm/examples"
if _examples_path not in sys.path:
    sys.path.insert(0, _examples_path)
from robot_controllers import GVController

TOPIC_NODE_ID = os.getenv("TOPIC_NODE_ID", "0_283")

gv_controller = None
running = True


def handle_command(cmd: dict) -> dict:
    """处理 cmd_vel 指令"""
    global gv_controller
    if gv_controller is None:
        return {"success": False, "error": "controller not initialized"}

    action = cmd.get("action", "move")
    linear_x = float(cmd.get("linear_x", 0.0))
    linear_y = float(cmd.get("linear_y", 0.0))
    angular_z = float(cmd.get("angular_z", 0.0))
    duration = float(cmd.get("duration_s", 1.0))

    if action == "stop":
        gv_controller.publish_velocity_command(0.0, 0.0, 0.0)
        return {"success": True, "action": "stopped"}

    # 设置 VR 模式为关闭 (允许直接 cmd_vel)
    gv_controller.set_vr_control_mode(enabled=False)
    gv_controller.publish_velocity_command(linear_x, linear_y, angular_z)
    time.sleep(duration)
    gv_controller.publish_velocity_command(0.0, 0.0, 0.0)

    return {
        "success": True,
        "action": action,
        "linear_x": linear_x,
        "angular_z": angular_z,
        "duration_s": duration,
    }


def stdin_loop():
    """主循环: 从 stdin 读取 JSON 指令"""
    global running
    print("cmd_vel_bridge ready", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
            result = handle_command(cmd)
            print(json.dumps(result), flush=True)
        except json.JSONDecodeError as e:
            print(json.dumps({"success": False, "error": f"invalid json: {e}"}), flush=True)
        except Exception as e:
            print(json.dumps({"success": False, "error": str(e)}), flush=True)


def main():
    global gv_controller, running

    rclpy.init()

    gv_controller = GVController(TOPIC_NODE_ID=TOPIC_NODE_ID)

    executor = MultiThreadedExecutor()
    executor.add_node(gv_controller)

    if not gv_controller.wait_for_initialization():
        print(json.dumps({"success": False, "error": "GVController init timeout"}), flush=True)
        return 1

    # Executor 线程
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # stdin 指令循环
    try:
        stdin_loop()
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        gv_controller.publish_velocity_command(0.0, 0.0, 0.0)
        executor.shutdown()
        gv_controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
