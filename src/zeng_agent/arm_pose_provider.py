from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EefPose:
    """末端执行器位姿"""
    position: list[float]  # [x, y, z] in meters (robot frame)
    rotation: list[float]  # [qx, qy, qz, qw] quaternion


@dataclass(slots=True)
class ArmPoseSnapshot:
    """全臂姿态快照"""
    timestamp: float
    left_eef: EefPose
    right_eef: EefPose
    head_pose: dict[str, list[float]] | None = None
    raw: dict[str, Any] | None = None


def parse_eef_pose(raw_json: str | dict[str, Any]) -> ArmPoseSnapshot:
    """从 ROS2 topic JSON 字符串解析手臂姿态

    输入格式（来自 topic_arm_current_robot_eef_pose_0_283）:
        {"timestamp": 1234.56,
         "left_eef_pose": {"position": [x,y,z], "rotation": [qx,qy,qz,qw]},
         "right_eef_pose": {"position": [x,y,z], "rotation": [qx,qy,qz,qw]},
         "head_pose": {"position": [x,y,z], "rotation": [qx,qy,qz,qw]}}

    返回: ArmPoseSnapshot
    """
    if isinstance(raw_json, str):
        data = json.loads(raw_json)
    else:
        data = raw_json

    return ArmPoseSnapshot(
        timestamp=data.get("timestamp", 0.0),
        left_eef=EefPose(
            position=list(data["left_eef_pose"]["position"]),
            rotation=list(data["left_eef_pose"]["rotation"]),
        ),
        right_eef=EefPose(
            position=list(data["right_eef_pose"]["position"]),
            rotation=list(data["right_eef_pose"]["rotation"]),
        ),
        head_pose=data.get("head_pose"),
        raw=data,
    )


class ArmPoseProvider:
    """手臂姿态读取器

    通过 ROS2 topic 获取机械臂当前末端执行器位姿。

    使用方式 (非 ROS2 环境):
        provider = ArmPoseProvider(ros_topic="/topic_arm_current_robot_eef_pose_0_283")
        snapshot = provider.get_pose()  # 返回 ArmPoseSnapshot

    使用方式 (在 ROS2 node 内):
        provider = ArmPoseProvider(node=rclpy_node)
        snapshot = provider.get_pose()

    注意: ArmPoseProvider 创建自己的 rclpy node,
          调用完成后需调用 shutdown() 清理。
    """

    def __init__(
        self,
        topic_node_id: str = "0_283",
        ros_topic: str | None = None,
        timeout: float = 2.0,
    ) -> None:
        self._topic = ros_topic or f"/topic_arm_current_robot_eef_pose_{topic_node_id}"
        self._timeout = timeout
        self._node = None
        self._rclpy_inited = False

    def get_pose(self) -> ArmPoseSnapshot | None:
        """获取当前手臂姿态（同步调用）

        返回 ArmPoseSnapshot 或 None（超时/失败）
        """
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String

        if not rclpy.ok():
            rclpy.init()
            self._rclpy_inited = True

        self._node = Node("arm_pose_provider")
        result: list[ArmPoseSnapshot] = []

        def cb(msg: String) -> None:
            try:
                result.append(parse_eef_pose(msg.data))
            except (KeyError, json.JSONDecodeError, ValueError) as e:
                self._node.get_logger().warn(f"Failed to parse pose: {e}")

        sub = self._node.create_subscription(String, self._topic, cb, 1)

        import time
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline and not result:
            rclpy.spin_once(self._node, timeout_sec=0.05)

        self._node.destroy_node()

        if not result:
            return None

        return result[0]

    def shutdown(self) -> None:
        import rclpy

        if self._rclpy_inited and rclpy.ok():
            rclpy.shutdown()

    def __enter__(self) -> ArmPoseProvider:
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()
