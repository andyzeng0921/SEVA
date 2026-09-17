"""Read-only ROS evidence for the existing 306 mapping session."""
import argparse
import collections
import json
import math
from pathlib import Path
import time

import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan, BatteryState
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
from tf2_ros import Buffer, TransformListener


def angle(q):
    return math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2 * (q.y*q.y + q.z*q.z))


def graph_summary(msg):
    # 2.8.3 publishes edge endpoint positions, not edge IDs or closure events.
    nodes = {m.id: [m.pose.position.x, m.pose.position.y]
             for m in msg.markers if m.ns == 'slam_toolbox' and m.action == 0}
    # Index observed endpoint positions; leave SLAM and its graph unchanged.
    # Adjacent cells preserve the original strict distance / unique-match rule.
    tolerance = 1e-7
    cells = collections.defaultdict(list)
    for node_id, xy in nodes.items():
        if all(math.isfinite(v) for v in xy):
            cells[(math.floor(xy[0]/tolerance), math.floor(xy[1]/tolerance))].append(node_id)

    def endpoint_id(point):
        if not math.isfinite(point.x) or not math.isfinite(point.y):
            return None
        cx, cy = math.floor(point.x/tolerance), math.floor(point.y/tolerance)
        matches = [i for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                   for i in cells.get((cx+dx, cy+dy), ())
                   if math.hypot(nodes[i][0]-point.x, nodes[i][1]-point.y) < tolerance]
        return matches[0] if len(matches) == 1 else None

    edges = []
    for m in msg.markers:
        if m.ns != 'slam_toolbox_edges' or m.action != 0:
            continue
        for a, b in zip(m.points[::2], m.points[1::2]):
            ends = [endpoint_id(point) for point in (a, b)]
            edges.append({'node_ids_from_positions': ends,
                          'xy': [[a.x, a.y], [b.x, b.y]],
                          'localization_edge': m.id == 1})
    return {'node_count': len(nodes), 'edge_count': len(edges), 'nodes': nodes,
            'edges': edges, 'loop_closure_verified': False,
            'note': 'Edge count or revisit alone does not prove a loop closure.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('output', type=Path)
    p.add_argument('--seconds', type=float, default=8.)
    args = p.parse_args()
    rclpy.init()
    n = rclpy.create_node('zeng306_readonly_session_health')
    buf = Buffer()
    listener = TransformListener(buf, n)
    state, received, counts = {}, {}, collections.Counter()

    def receive(key, msg):
        state[key] = msg
        received[key] = time.monotonic()
        counts[key] += 1

    specs = [('front', LaserScan, '/topic_gv_front_lidar_0_306', qos_profile_sensor_data),
             ('rear', LaserScan, '/topic_gv_rear_lidar_0_306', qos_profile_sensor_data),
             ('front_filtered', LaserScan, '/zeng306/front_filtered', qos_profile_sensor_data),
             ('rear_filtered', LaserScan, '/zeng306/rear_filtered', qos_profile_sensor_data),
             ('merged', LaserScan, '/zeng306/mapping_scan', qos_profile_sensor_data),
             ('odom', Odometry, '/topic_gv_wheel_odom_0_306', qos_profile_sensor_data),
             ('battery', BatteryState, '/topic_gv_battery_0_306', qos_profile_sensor_data),
             ('robot_state', String, '/robot_state_topic_0_306', 10),
             ('graph', MarkerArray, '/slam_toolbox/graph_visualization', 10),
             ('map', OccupancyGrid, '/map', QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))]
    for key, typ, topic, qos in specs:
        n.create_subscription(typ, topic, lambda m, k=key: receive(k, m), qos)
    started = time.monotonic()
    while time.monotonic()-started < args.seconds:
        rclpy.spin_once(n, timeout_sec=.025)
    out = {'time': time.time(), 'duration_s': time.monotonic()-started,
           'read_only': True, 'counts': dict(counts), 'sensors': {}, 'transforms': {}}
    for key, msg in state.items():
        age = (n.get_clock().now()-rclpy.time.Time.from_msg(msg.header.stamp)).nanoseconds/1e9 if hasattr(msg, 'header') else None
        if isinstance(msg, LaserScan):
            valid = [r for r in msg.ranges if math.isfinite(r) and msg.range_min <= r <= msg.range_max]
            out['sensors'][key] = {'header_age_s': age, 'received_age_s': time.monotonic()-received[key],
                                   'frame': msg.header.frame_id, 'beam_count': len(msg.ranges),
                                   'valid_count': len(valid), 'min_range_m': min(valid) if valid else None}
        elif isinstance(msg, Odometry):
            out['odom'] = {'pose': [msg.pose.pose.position.x, msg.pose.pose.position.y, angle(msg.pose.pose.orientation)],
                           'velocity': [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z], 'header_age_s': age}
        elif isinstance(msg, BatteryState):
            out['battery'] = {'voltage': msg.voltage, 'current': msg.current, 'percentage': msg.percentage,
                              'status': msg.power_supply_status}
        elif isinstance(msg, OccupancyGrid):
            c = collections.Counter(msg.data)
            out['map'] = {'width': msg.info.width, 'height': msg.info.height, 'resolution': msg.info.resolution,
                          'free_cells': c[0], 'occupied_cells': c[100], 'unknown_cells': c[-1],
                          'observed_free_m2': c[0] * msg.info.resolution**2,
                          'environment_coverage_fraction': None}
        elif key == 'graph':
            out['graph'] = graph_summary(msg)
        elif key == 'robot_state':
            try:
                data = json.loads(msg.data)
                out['robot_state'] = {k: data.get(k) for k in ['timestamp', 'state', 'joints_pos', 'self_collide', 'protection']}
            except ValueError:
                out['robot_state'] = {'error': 'invalid_json'}
    for target, source in [('map', 'base_link'), ('map', 'odom')]:
        try:
            t = buf.lookup_transform(target, source, rclpy.time.Time())
            out['transforms'][target+'_'+source] = {'pose': [t.transform.translation.x, t.transform.translation.y, angle(t.transform.rotation)],
                'header_age_s': (n.get_clock().now()-rclpy.time.Time.from_msg(t.header.stamp)).nanoseconds/1e9}
        except Exception as exc:
            out['transforms'][target+'_'+source] = {'error': str(exc)}
    out['publishers'] = {topic: [{'node': a.node_name, 'namespace': a.node_namespace, 'gid': list(a.endpoint_gid)}
                                for a in n.get_publishers_info_by_topic(topic)]
                         for topic in ['/map', '/zeng306/mapping_scan', '/topic_gv_target_cmd_vel_0_306']}
    out['nodes'] = n.get_node_names_and_namespaces()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k not in ['graph', 'nodes', 'robot_state']}, ensure_ascii=False))
    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
