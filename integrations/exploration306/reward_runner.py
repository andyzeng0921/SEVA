"""Session orchestration using native candidate, planning and navigation APIs."""
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav2_msgs.srv import IsPathValid
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_srvs.srv import Trigger

from .reward_policy import (Experience, GraphGain, MapGain, history_recovery_candidates, load_semantics,
    observed_reward, revisit_candidates, semantic_novelty, write_json)

BRIDGE = os.environ.get('SEVA_FRONTIER_BRIDGE', str(Path(__file__).resolve().parents[2] /
    'ros2_ws/install/zeng306_frontier_bridge/lib/zeng306_frontier_bridge/candidate_bridge'))


def pose_msg(node, xy, yaw):
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = node.get_clock().now().to_msg()
    pose.pose.position.x, pose.pose.position.y = map(float, xy)
    pose.pose.orientation.z, pose.pose.orientation.w = math.sin(yaw/2), math.cos(yaw/2)
    return pose


class NativeCandidates:
    """No navigation/velocity interface: safe to use for isolated replay tests."""
    def __init__(self, node, wait):
        self.node, self.wait = node, wait
        self.source = node.create_client(Trigger, '/zeng306/frontier_candidates')
        self.planner = ActionClient(node, ComputePathToPose, '/compute_path_to_pose')
        self.validator = node.create_client(IsPathValid, '/is_path_valid')
        self.grid = None
        self.robot_pose = None
        self.map_subscription = node.create_subscription(OccupancyGrid, '/map', self.receive_map,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def receive_map(self, msg):
        self.grid = msg

    def known_free(self, xy):
        grid = self.grid
        if grid is None or grid.header.frame_id != 'map':
            return False
        info = grid.info
        x = math.floor((xy[0]-info.origin.position.x)/info.resolution)
        y = math.floor((xy[1]-info.origin.position.y)/info.resolution)
        return 0 <= x < info.width and 0 <= y < info.height and grid.data[y*info.width+x] == 0

    def snapshot(self):
        if not self.source.wait_for_service(timeout_sec=2):
            raise RuntimeError('native_candidate_service_missing')
        result = self.wait(self.source.call_async(Trigger.Request()), 12)
        if not result.success:
            raise RuntimeError('native_candidates:' + result.message)
        snapshot = json.loads(result.message)
        self.robot_pose = snapshot['pose']
        return snapshot

    def validate(self, candidate):
        result = self._validate(candidate)
        if (result['path_valid'] or candidate['kind'] != 'frontier'
                or candidate.get('observation_standoff_applied') or self.robot_pose is None
                or result['planning_reason'] not in ('no_native_path', 'native_path_invalid')):
            return result
        # The raw frontier lies at a sensing boundary. Probe a bounded set of
        # observation goals on its known side using the unchanged native planner.
        fx, fy = candidate['xy']
        heading = math.atan2(self.robot_pose[1]-fy, self.robot_pose[0]-fx)
        attempts = []
        for offset in (0.8, 1.2):
            for delta in (0., -math.pi/4, math.pi/4):
                xy = [fx+offset*math.cos(heading+delta), fy+offset*math.sin(heading+delta)]
                alternative = {**candidate, 'source_frontier_xy': candidate['xy'], 'xy': xy,
                    'yaw': math.atan2(fy-xy[1], fx-xy[0]), 'observation_standoff_applied': True,
                    'observation_offset_m': offset, 'native_observation_endpoint_allowed': True,
                    'gain_reference': 'upstream frontier visibility proxy; actual gain measured after arrival'}
                checked = self._validate(alternative)
                attempts.append({'xy': xy, 'planning_reason': checked['planning_reason']})
                if checked['path_valid']:
                    return {**checked, 'observation_probe_attempts': attempts}
        return {**result, 'observation_probe_attempts': attempts}

    def _validate(self, candidate):
        if not self.known_free(candidate['xy']):
            return {**candidate, 'path_valid': False, 'planning_reason': 'goal_not_in_observed_free_space'}
        if not self.planner.wait_for_server(timeout_sec=2) or not self.validator.wait_for_service(timeout_sec=2):
            raise RuntimeError('native_planning_or_validation_api_missing')
        goal = ComputePathToPose.Goal()
        goal.goal = pose_msg(self.node, candidate['xy'], candidate['yaw'])
        goal.use_start = False
        goal.planner_id = 'GridBased'
        handle = self.wait(self.planner.send_goal_async(goal), 4)
        if not handle.accepted:
            return {**candidate, 'path_valid': False, 'planning_reason': 'rejected'}
        try:
            result = self.wait(handle.get_result_async(), 8)
        except BaseException:
            handle.cancel_goal_async()
            raise
        path = result.result.path
        if result.status != 4 or result.result.error_code != 0 or len(path.poses) < 2:
            return {**candidate, 'path_valid': False, 'planning_reason': 'no_native_path',
                    'planning_error_code': result.result.error_code}
        endpoint = path.poses[-1].pose.position
        endpoint_xy = [endpoint.x, endpoint.y]
        endpoint_error = math.dist(endpoint_xy, candidate['xy'])
        if candidate.get('native_observation_endpoint_allowed') and endpoint_error > .15:
            # Smac may return a nearby endpoint within its configured tolerance.
            # Make that explicit observation goal and replan it, rather than
            # relaxing the endpoint or footprint acceptance rule.
            if not .15 < endpoint_error <= .31 or not self.known_free(endpoint_xy):
                return {**candidate, 'path_valid': False, 'planning_reason': 'native_observation_endpoint_rejected'}
            fx, fy = candidate['source_frontier_xy']
            adjusted = {**candidate, 'xy': endpoint_xy, 'yaw': math.atan2(fy-endpoint.y, fx-endpoint.x),
                'native_observation_endpoint_allowed': False,
                'native_endpoint_adjustment_m': endpoint_error,
                'native_endpoint_replanned': True}
            return self._validate(adjusted)
        if candidate['kind'] == 'frontier' and not candidate.get('observation_standoff_applied'):
            # Choose an observation pose 0.8 m back along the native computed path.
            # This is goal placement glue, not a replacement path planner.
            remaining = 0.0
            index = len(path.poses)-1
            while index > 0 and remaining < 0.8:
                a, b = path.poses[index-1].pose.position, path.poses[index].pose.position
                remaining += math.hypot(b.x-a.x, b.y-a.y)
                index -= 1
            point = path.poses[index].pose.position
            xy = [point.x, point.y]
            yaw = math.atan2(candidate['xy'][1]-xy[1], candidate['xy'][0]-xy[0])
            candidate = {**candidate, 'source_frontier_xy':candidate['xy'], 'xy':xy, 'yaw':yaw,
                         'observation_standoff_applied':True, 'native_path_standoff_m':remaining,
                         'gain_reference':'upstream source frontier visibility proxy, before observation standoff'}
            path.poses = path.poses[:index+1]
            path.poses[-1] = pose_msg(self.node, xy, yaw)
        # Include the requested final orientation in native full-footprint validation.
        path.poses.append(pose_msg(self.node, candidate['xy'], candidate['yaw']))
        checked = self.wait(self.validator.call_async(IsPathValid.Request(path=path)), 4)
        xy = [[p.pose.position.x, p.pose.position.y] for p in path.poses]
        finite = all(math.isfinite(v) for point in xy for v in point)
        length = sum(math.dist(a,b) for a,b in zip(xy,xy[1:])) if finite else 0.0
        # Smac's tolerance may return a nearby endpoint; reject a substantial miss.
        endpoint_ok = finite and math.dist(xy[-2], candidate['xy']) <= 0.15
        known_path = finite and all(self.known_free(point) for point in xy)
        return {**candidate, 'path_valid': bool(checked.is_valid and endpoint_ok and known_path),
                'path_length_m': length, 'path_points': len(xy),
                'planning_reason': 'native_path_validated' if checked.is_valid and endpoint_ok and known_path else 'native_path_invalid',
                'invalid_pose_count': len(checked.invalid_pose_indices),
                'endpoint_error_m': math.dist(xy[-2], candidate['xy']) if finite else None,
                'path_centers_observed_free': known_path}


def map_gain(ledger, msg, epoch):
    q = msg.info.origin.orientation
    yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
    return ledger.update(np.asarray(msg.data).reshape(msg.info.height, msg.info.width),
        msg.info.resolution, [msg.info.origin.position.x, msg.info.origin.position.y], epoch, yaw)


def features(observations):
    return {feature for o in observations if o.get('schema_valid')
            and o.get('semantic_scene', {}).get('view_quality') == 'good'
            for feature in o['semantic_scene'].get('place_features', [])}


def run_reward_session(session, seconds):
    s = session
    if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'zeng306-coverage-navigation.service']).returncode:
        raise RuntimeError('owned_navigation_unit_not_active')
    s.pump(4)
    s.check()
    if not s.parked() or any(g.status in [1,2,3] for g in getattr(s.state.get('nav'), 'status_list', [])):
        raise RuntimeError('robot_not_idle')
    if not s.navigate.wait_for_server(timeout_sec=2):
        raise RuntimeError('navigation_unavailable')
    epoch = bytes(s.owner[0]).hex()
    policy_config = json.loads((s.folder/'config/reward.306.json').read_text())
    memory = Experience(epoch, s.folder/'reward/experience.json', policy_config)
    area_ledger, graph_ledger = MapGain(), GraphGain()
    with s.feedback_lock:
        s.origin = s.last_xy[:]
        s.distance = 0.0
        s.moving_session = True
        graph = s.graphs[-1] if s.graphs else {}
        initial_map = s.state['map']
    map_gain(area_ledger, initial_map, epoch)
    graph_ledger.update(graph, epoch)
    s.bag = s.spawn(['ros2', 'bag', 'record', '-s', 'mcap', '-o', str(s.folder/'bags'/s.evidence.name),
        '/tf', '/tf_static', '/map', '/zeng306/mapping_scan', '/topic_gv_front_lidar_0_306',
        '/topic_gv_rear_lidar_0_306', '/topic_gv_wheel_odom_0_306', '/topic_gv_target_cmd_vel_0_306',
        '/slam_toolbox/graph_visualization'], s.evidence.name+'-bag.log')
    s.checkpoint('coverage-start')
    s.native = s.spawn([BRIDGE], s.evidence.name+'-candidates.log')
    native = NativeCandidates(s.node, s.wait)
    s.native_candidate_api = native  # Retain ROS clients until the session executor stops.
    s.deadline = time.monotonic()+seconds
    last_revisit_distance, empty_rounds, pending = 0.0, 0, None
    history_recoveries = 0
    s.pump(4, guarded=True)
    try:
        for round_id in range(memory.config['maximum_goals']):
            s.phase = 'reward_selection'
            s.check()
            if s.native.poll() is not None:
                raise RuntimeError('native_candidate_bridge_exited')
            s.reacquire_stationary_feedback()
            snapshot = native.snapshot()
            with s.feedback_lock:
                graph = s.graphs[-1] if s.graphs else {}
            observations = load_semantics(s.folder/'semantics')
            candidates = snapshot['candidates'] + revisit_candidates(graph, snapshot['pose'],
                s.distance-last_revisit_distance, memory.config)
            checked = []
            for c in candidates:
                s.check()
                if memory.blocked(c['xy'], time.time()):
                    checked.append({**c, 'path_valid': False, 'planning_reason': 'failure_cooldown'})
                    continue
                c['semantic_novelty'] = semantic_novelty(c, observations, graph, epoch)
                c = native.validate(c)
                checked.append(c)
                if not c['path_valid']:
                    memory.record({'time': time.time(), 'xy': c['xy'], 'kind': c['kind'], 'success': False,
                        'reason': c['planning_reason'], 'motion_issued': False, 'new_known_m2': 0})
            ranked = memory.rank(checked, time.time())
            if not ranked and history_recoveries < memory.config['history_recovery_max_per_run']:
                # A restarted run has zero new mileage but may contain a valid
                # long native history. Probe a single short reconnect opportunity.
                with s.feedback_lock:
                    graph = s.graphs[-1] if s.graphs else {}
                for c in history_recovery_candidates(graph, snapshot['pose'], memory.config):
                    s.check()
                    if memory.blocked(c['xy'], time.time()):
                        continue
                    c['semantic_novelty'] = semantic_novelty(c, observations, graph, epoch)
                    c = native.validate(c)
                    checked.append(c)
                ranked = memory.rank(checked, time.time())
            decision = {'time': time.time(), 'map_epoch': epoch, 'upstream_snapshot': snapshot,
                        'checked_candidates': checked, 'ranked': ranked,
                        'selected': ranked[0] if ranked else None,
                        'semantic_model': 'qwen3.5-35b-a3b', 'semantic_records': len(observations)}
            write_json(s.evidence/f'decision-{round_id:03}.json', decision)
            write_json(s.folder/'reward/latest-decision.json', decision)
            if not ranked:
                empty_rounds += 1
                s.event('reward_no_reachable_candidates', native_frontiers=snapshot['native_frontier_count'], round=empty_rounds)
                if empty_rounds >= memory.config['maximum_empty_rounds']:
                    return 'no_reachable_reward_candidates_requires_coverage_review'
                s.capture()
                s.pump(5, guarded=True)
                continue
            empty_rounds = 0
            selected = ranked[0]
            # Revalidate against fresh native costmap immediately before dispatch.
            if not native.validate(selected)['path_valid']:
                memory.record({'time':time.time(),'xy':selected['xy'],'kind':selected['kind'],
                    'success':False,'reason':'path_changed_before_dispatch','motion_issued':False,'new_known_m2':0})
                continue
            s.check()
            if selected['kind'] == 'revisit_recovery':
                history_recoveries += 1
            pending = {'time': time.time(), 'xy': selected['xy'], 'kind': selected['kind'],
                       'success': False, 'motion_issued': True, 'new_known_m2': 0}
            distance_before = s.distance
            before_features = features(observations)
            s.phase = 'reward_' + selected['kind']
            s.event('reward_goal_selected', goal=selected)
            goal = NavigateToPose.Goal()
            goal.pose = pose_msg(s.node, selected['xy'], selected['yaw'])
            # Existing native tree replans while the controller and Collision Monitor protect motion.
            goal.behavior_tree = str(s.folder/'config/native-navigation-reward.xml')
            s.return_handle = s.wait(s.navigate.send_goal_async(goal), 5)
            if not s.return_handle.accepted:
                raise RuntimeError('reward_navigation_rejected')
            result = s.wait(s.return_handle.get_result_async(), min(240, 60+selected['path_length_m']/0.04))
            s.return_handle = None
            pending.update(native_status=result.status, native_error_code=getattr(result.result, 'error_code', None))
            if result.status != 4:
                raise RuntimeError('reward_native_navigation_failed_' + str(result.status))
            s.pump(2, guarded=True)
            s.reacquire_stationary_feedback()
            actual = s.transform('base_link').transform.translation
            error = math.dist([actual.x, actual.y], selected['xy'])
            if error > 0.4:
                raise RuntimeError('reward_goal_position_not_verified')
            s.capture()
            until = min(s.deadline-1, time.monotonic()+45)
            while s.semantic.poll() is None and time.monotonic() < until:
                s.pump(.1, guarded=True)
            with s.feedback_lock:
                grid = s.state['map']
                graph = s.graphs[-1] if s.graphs else {}
            gain = map_gain(area_ledger, grid, epoch)
            graph_gain = graph_ledger.update(graph, epoch)
            new_features = len(features(load_semantics(s.folder/'semantics'))-before_features)
            travelled = s.distance-distance_before
            outcome = observed_reward(gain['new_known_m2'], graph_gain, new_features, travelled, True)
            pending.update(success=True, reason='verified_native_goal', position_error_m=error,
                new_constraints=graph_gain, new_visual_features=new_features, distance_m=travelled,
                **gain, **outcome)
            memory.record(pending)
            s.event('reward_goal_result', outcome=pending)
            pending = None
            if selected['kind'] in ('revisit', 'revisit_recovery'):
                last_revisit_distance = s.distance
            write_json(s.evidence/'live.json', {'phase':s.phase,'distance_m':s.distance,
                'reward':outcome,'new_known_m2':gain['new_known_m2'],'new_cross_visit_constraints':graph_gain})
            if round_id % 3 == 2 or selected['kind'] in ('revisit', 'revisit_recovery'):
                s.checkpoint(f'reward-{round_id:03}')
            if s.deadline-time.monotonic() < 30:
                return 'reward_session_budget_finished'
        return 'reward_goal_budget_finished'
    finally:
        if pending is not None:
            pending.update(reason='navigation_or_monitor_interrupted',
                           **observed_reward(0, 0, 0, s.distance-distance_before, False))
            memory.record(pending)
            s.event('reward_goal_interrupted', outcome=pending)
