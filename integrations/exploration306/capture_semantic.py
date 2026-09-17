"""Session glue: existing RGB-D reader, ROS TF, and existing Qwen annotation adapter."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import rclpy
from rclpy.time import Time
from rosidl_runtime_py.convert import message_to_ordereddict
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import MarkerArray
from upstream306.observations import capture_rgbd
from upstream306.semantic_map import annotate_viewpoint
from zeng_agent.config import AgentConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('session', type=Path)
    args = parser.parse_args()
    rclpy.init()
    node = rclpy.create_node('zeng_306_semantic_viewpoint')
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    graph = {}
    node.create_subscription(MarkerArray, '/slam_toolbox/graph_visualization',
        lambda msg: graph.update(message=msg, received=time.monotonic()), 1)
    try:
        until = time.monotonic() + 2.5
        while time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=.05)
        observation = capture_rgbd(args.session / 'observations')
        if not observation.success:
            raise RuntimeError(observation.message)
        data = observation.data
        until = time.monotonic() + .6
        while time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=.05)
        capture_ns = sum(data['producer_timestamps_ns']) // 2
        pose = buffer.lookup_transform('map', 'base_link', Time(nanoseconds=capture_ns))
        anchor = None
        owners = node.get_publishers_info_by_topic('/map')
        if graph and len(owners) == 1 and time.monotonic()-graph['received'] < 3.0:
            points = [m for m in graph['message'].markers if m.ns == 'slam_toolbox' and m.action == 0 and m.id >= 188]
            xy = [pose.transform.translation.x, pose.transform.translation.y]
            if points:
                nearest = min(points, key=lambda m: math.dist(xy, [m.pose.position.x,m.pose.position.y]))
                separation = math.dist(xy, [nearest.pose.position.x,nearest.pose.position.y])
                if separation <= 0.8:
                    anchor = {'node_id': nearest.id, 'map_epoch': bytes(owners[0].endpoint_gid).hex(),
                              'viewpoint_distance_to_node_m': separation,
                              'binding': 'nearby native SLAM node; viewpoint vicinity, not object position'}
        observation_id = data['observation_id']
        image_path = args.session / 'observations' / (observation_id + '.jpg')
        with np.load(data['artifact']) as frames:
            Image.fromarray(frames['rgb']).save(image_path, quality=95)
        state = {'observation_id': observation_id, 'captured_at': capture_ns / 1e9,
                 'slam_pose': message_to_ordereddict(pose),
                 'graph_anchor': anchor,
                 'camera': asdict(observation), 'image': str(image_path),
                 'pose_binding': 'map to base_link at RGB-D capture timestamp',
                 'object_world_coordinates': None}
        (args.session / 'observations' / (observation_id + '.json')).write_text(
            json.dumps(state, ensure_ascii=False, indent=2))
        print(json.dumps({'captured': observation_id, 'image': str(image_path),
                          'slam_pose': state['slam_pose']}, ensure_ascii=False), flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    config = AgentConfig.from_yaml(os.environ['ZENG_CONFIG'])
    record = annotate_viewpoint(image_path, state, config.fire_search,
        args.session / 'semantics' / (observation_id + '.json'))
    print(json.dumps({'observation_id': observation_id,
                      'schema_valid': record['schema_valid'],
                      'scene': record.get('semantic_scene'),
                      'error': record.get('validation_error') or record['response'].get('error')},
                     ensure_ascii=False), flush=True)
    if not record['schema_valid']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
