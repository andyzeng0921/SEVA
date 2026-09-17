#!/usr/bin/env python3
"""
安全关节标定 v2 — 只测身体前方角度

碰撞根因:
  Shoulder_Inner 正值 → 手臂甩到身后 (X 负) → 撞身体
安全策略:
  - Shoulder_Inner 只用负值 (手臂向前)
  - 所有测试保持 X >= -0.05
  - 每次测试前后拍照对比
"""
import json, os, time, sys
os.environ["LD_LIBRARY_PATH"] = "/opt/robot/env/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

NID = "0_283"
SAFE_X = -0.05

def make_msg(la, ra):
    return json.dumps({
        "leg_waist_target_joints_position": [0.0]*4,
        "left_arm_target_joints_position": la,
        "right_arm_target_joints_position": ra,
    })

def get_pose(node, topic, to=5.0):
    r = []
    s = node.create_subscription(String, topic, lambda m: r.append(json.loads(m.data)), 1)
    dl = time.monotonic() + to
    while time.monotonic() < dl and not r:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_subscription(s)
    return r[0] if r else None

def cmd(node, la, ra, wait=3.0):
    pub = node.create_publisher(String, f"/topic_arm_whole_body_target_joints_position_{NID}", 1)
    time.sleep(0.1)
    pub.publish(String(data=make_msg(la, ra)))
    node.destroy_publisher(pub)
    time.sleep(wait)

def snap(node, label):
    p = get_pose(node, f"/topic_arm_current_robot_eef_pose_{NID}")
    if p:
        lp = p["left_eef_pose"]["position"]
        rp = p["right_eef_pose"]["position"]
        danger = ""
        if lp[0] < SAFE_X or rp[0] < SAFE_X:
            danger = " ⚠⚠⚠ 危险!到身后!"
        print(f"  {label:28s} 左X={lp[0]:+.3f} Y={lp[1]:+.3f} Z={lp[2]:.3f}  右X={rp[0]:+.3f} Y={rp[1]:+.3f} Z={rp[2]:.3f}{danger}")
        return p
    return None

def do_reset(node):
    for topic, data in [
        (f"/control_reset_{NID}", String()),
        (f"/topic_arm_robot_action_{NID}", String(data=json.dumps({"action_type":"play","action_name":"reset"}))),
        (f"/topic_arm_move_up_down_z_{NID}", String(data=json.dumps({"height_z":0.0,"move_joint_waist_pitch":True}))),
    ]:
        pub = node.create_publisher(String, topic, 1)
        time.sleep(0.1)
        pub.publish(data)
        node.destroy_publisher(pub)
    time.sleep(8)

def main():
    rclpy.init()
    n = Node("safe_calib2")
    print("="*70)
    print("安全标定 v2 — 只测身体前方 (X>=-0.05=安全)")

    Z = [0.0]*7  # 零位向量

    # ── 测试组 1: 左臂 Shoulder_Inner 负值(前抬) ──
    print("\n═══ T1: 左臂 Shoulder_Inner 负值 (前抬) ═══")
    for a in [-10, -20, -30, -45]:
        do_reset(n)
        cmd(n, [float(a),0,0,0,0,0,0], Z)
        snap(n, f"L_Inner={a:+3d}°")

    # ── 测试组 2: 左臂 UpperArm 正值 (前抬) ──
    print("\n═══ T2: 左臂 UpperArm 正值 ═══")
    for a in [20, 30, 50, 70]:
        do_reset(n)
        cmd(n, [0,0,float(a),0,0,0,0], Z)
        snap(n, f"L_UpperArm={a:+3d}°")

    # ── 测试组 3: 左臂组合 (Inner负 + UpperArm正) ──
    print("\n═══ T3: 左臂 Inner(-) + UpperArm(+) 组合 ═══")
    for (i, u) in [(-20,30), (-30,30), (-40,30), (-20,50)]:
        do_reset(n)
        cmd(n, [float(i),0,float(u),0,0,0,0], Z)
        snap(n, f"L_In={i:+3d}°+UA={u:+3d}°")

    # ── 测试组 4: 右臂对称 (Inner正, UpperArm负) ──
    print("\n═══ T4: 右臂 Shoulder_Inner 正值(前抬) ═══")
    for a in [10, 20, 30]:
        do_reset(n)
        cmd(n, Z, [float(a),0,0,0,0,0,0])
        snap(n, f"R_Inner={a:+3d}°")

    print("\n═══ T5: 右臂组合 (Inner正 + UpperArm负) ═══")
    for (i, u) in [(20,-30), (30,-30), (30,-50)]:
        do_reset(n)
        cmd(n, Z, [float(i),0,float(u),0,0,0,0])
        snap(n, f"R_In={i:+3d}°+UA={u:+3d}°")

    # ── 交叉测试 ──
    print("\n═══ T6: 左手上 + 右手下 交叉 ═══")
    do_reset(n)
    cmd(n, [-30,0,30,0,0,0,0], Z, wait=5)
    snap(n, "Phase1: L↑ R↓")

    print("\n═══ T7: 左手下 + 右手上 交叉 ═══")
    cmd(n, Z, [30,0,-30,0,0,0,0], wait=5)
    snap(n, "Phase2: L↓ R↑")

    cmd(n, Z, Z, wait=3)
    snap(n, "复位")

    do_reset(n)
    print("\n✅ 安全标定完成")
    n.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
