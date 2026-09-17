"""Supervise upstream explore_lite; revisit using native Nav2 actions.

This file does not implement frontier search, path planning, control, or SLAM.
Default mode only collects evidence. --run requires the owned navigation stack
and an already undocked, physically prepared robot.
"""
import argparse
import collections
import copy
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
from action_msgs.msg import GoalStatusArray
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import PolygonStamped, PoseStamped, Twist
from nav_msgs.msg import Odometry, OccupancyGrid
from nav2_msgs.action import NavigateToPose, Spin
from sensor_msgs.msg import LaserScan, BatteryState
from std_msgs.msg import Bool, String
from visualization_msgs.msg import MarkerArray
from explore_lite_msgs.msg import ExploreStatus
from slam_toolbox.srv import SaveMap, SerializePoseGraph
from tf2_ros import Buffer, TransformListener
from session_health import angle, graph_summary

EXPLORER = os.environ.get('SEVA_LEGACY_EXPLORER', '/opt/robot/explore/explore_lite/lib/explore_lite/explore')
NAV_UNIT = os.environ.get('SEVA_NAV_UNIT', 'zeng306-coverage-navigation.service')


def command_matches_native(command, candidates):
    """Collision Monitor may preserve, scale down, or stop a native command."""
    if not all(math.isfinite(v) for v in command):
        return False
    for candidate in candidates:
        norm = sum(v*v for v in candidate)
        if norm < 1e-8:
            continue
        scale = sum(a*b for a,b in zip(command, candidate))/norm
        if -.001 <= scale <= 1.02 and max(abs(a-scale*b) for a,b in zip(command, candidate)) <= .002:
            return True
    return False


def fail_if_unhealthy(ages, velocity, battery_current, battery_status, distance, radius):
    """Fail closed on real telemetry, including source timestamps."""
    for key in ['front', 'rear', 'merged', 'odom']:
        arrival, header = ages.get(key, (float('inf'), float('inf')))
        if not all(math.isfinite(v) for v in [arrival, header]) or arrival > .65 or not -.3 <= header <= .65:
            raise RuntimeError('stale_or_missing_' + key)
    if not 0 <= ages.get('battery', (float('inf'),))[0] <= 3:
        raise RuntimeError('stale_battery')
    if not all(math.isfinite(v) for v in velocity) or abs(velocity[0]) > .13 or abs(velocity[1]) > .015 or abs(velocity[2]) > .13:
        raise RuntimeError('unexpected_odom_speed')
    if not math.isfinite(battery_current) or battery_current >= -.15 or battery_status == BatteryState.POWER_SUPPLY_STATUS_CHARGING:
        raise RuntimeError('charging_requires_native_undock')
    if not math.isfinite(distance+radius) or distance > 120 or radius > 25:
        raise RuntimeError('segment_distance_limit')


def stationary_feedback_reason(history, now, ros_now):
    """Full-rate odometry only; sparse report trajectories are not telemetry."""
    rows = [r for r in history if 0 <= now-r[0] < 1.2]
    if not rows or now-rows[-1][0] >= .3:
        return 'missing_recent_odom'
    if any(not all(math.isfinite(v) for v in r) for r in rows):
        return 'invalid_odom'
    if any(max(abs(v) for v in r[2:5]) >= .003 for r in rows):
        return 'motion_detected'
    if len(rows) < 5 or rows[-1][0]-rows[0][0] < .8:
        return 'insufficient_stationary_history'
    if any(b[0]-a[0] > .3 or b[1] <= a[1] for a, b in zip(rows, rows[1:])):
        return 'odom_gap_or_replayed_stamp'
    if not -.3 <= ros_now-rows[-1][1] < .3:
        return 'stale_odom_stamp'
    if any(not -.3 <= (ros_now-r[1])-(now-r[0]) < .3 for r in rows):
        return 'stale_odom_at_receipt'
    return 'ok'


class Session:
    def __init__(self, folder, run_id):
        self.folder = folder
        self.evidence = folder / 'evidence' / run_id
        self.evidence.mkdir(parents=True, exist_ok=False)
        self.maps = folder/'maps'/run_id
        self.maps.mkdir(parents=True, exist_ok=False)
        self.node = rclpy.create_node('zeng306_coverage_supervisor')
        self.buf = Buffer()
        self.listener = TransformListener(self.buf, self.node)
        self.state, self.times = {}, {}
        self.feedback_lock = threading.RLock()
        self.receiver_error = None
        self.feedback_gaps = {}
        self.trajectory, self.events, self.graphs = [], [], []
        self.graph_counts = []
        self.odom_feedback = collections.deque(maxlen=512)
        self.checkpoint_pose = None
        self.checkpoint_motion = False
        self.distance = 0.
        self.last_xy = None
        self.origin = None
        self.owner = None
        self.anchor = None
        self.map_odom = None
        self.last_graph_check = 0.
        self.last_health_check = 0.
        self.native = self.semantic = self.bag = None
        self.return_handle = None
        self.children_logs = []
        self.moving_session = False
        self.interrupted = False
        self.phase = 'preflight'
        self.external_command = None
        self.native_commands = collections.deque(maxlen=60)
        self.pending_commands = collections.deque(maxlen=60)
        cad = json.loads((folder/'evidence'/'current-cad-clearance-before-coverage.json').read_text())
        self.body_reference = cad['joint_state']
        self.started_wall = time.time()
        self.deadline = None
        latched = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        latest_sensor = copy.copy(qos_profile_sensor_data)
        latest_sensor.depth = 1
        specs = [('odom', Odometry, '/topic_gv_wheel_odom_0_306', latest_sensor),
                 ('joints', String, '/topic_arm_whole_body_and_gripper_current_joints_status_0_306', 1),
                 ('footprint', PolygonStamped, '/local_costmap/published_footprint', 10),
                 ('front', LaserScan, '/zeng306/front_filtered', latest_sensor),
                 ('rear', LaserScan, '/zeng306/rear_filtered', latest_sensor),
                 ('merged', LaserScan, '/zeng306/mapping_scan', latest_sensor),
                 ('battery', BatteryState, '/topic_gv_battery_0_306', latest_sensor),
                 ('nav', GoalStatusArray, '/navigate_to_pose/_action/status', latched),
                 ('explore', ExploreStatus, '/explore/status', latched),
                 ('map', OccupancyGrid, '/map', latched),
                 ('graph', MarkerArray, '/slam_toolbox/graph_visualization', 10)]
        for key, typ, topic, qos in specs:
            self.node.create_subscription(typ, topic, lambda m, k=key: self.receive(k, m), qos)
        self.node.create_subscription(Twist, '/topic_gv_target_cmd_vel_0_306', self.command, 10)
        self.node.create_subscription(Twist, '/zeng306/cmd_vel_smoothed', self.native_command, 10)
        self.pause = self.node.create_publisher(Bool, '/explore/resume', 10)
        self.zero = self.node.create_publisher(Twist, '/zeng306/cmd_vel_smoothed', 10)
        self.navigate = ActionClient(self.node, NavigateToPose, '/navigate_to_pose')
        self.spin_action = ActionClient(self.node, Spin, '/spin')
        self.cancel_nav = self.node.create_client(CancelGoal, '/navigate_to_pose/_action/cancel_goal')
        self.saver = self.node.create_client(SaveMap, '/slam_toolbox/save_map')
        self.serializer = self.node.create_client(SerializePoseGraph, '/slam_toolbox/serialize_map')
        # Task orchestration / file IO must not block the ROS feedback executor.
        self.receiver = SingleThreadedExecutor()
        self.receiver.add_node(self.node)
        def receive_loop():
            try:
                self.receiver.spin()
            except BaseException as exc:
                self.receiver_error = repr(exc)
        self.receiver_thread = threading.Thread(target=receive_loop, name='native-feedback', daemon=True)
        self.receiver_thread.start()

    def receive(self, key, msg):
        with self.feedback_lock:
            self.receive_locked(key, msg)

    def receive_locked(self, key, msg):
        now = time.monotonic()
        if key in self.times:
            gap = now-self.times[key]
            self.feedback_gaps[key] = max(gap, self.feedback_gaps.get(key, 0.))
        self.state[key], self.times[key] = msg, time.monotonic()
        if key == 'odom':
            xy = [msg.pose.pose.position.x, msg.pose.pose.position.y]
            yaw = angle(msg.pose.pose.orientation)
            velocity = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z]
            stamp = rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds/1e9
            self.odom_feedback.append([self.times[key], stamp, *velocity, *xy, yaw])
            if self.checkpoint_pose is not None:
                delta_yaw = abs(math.atan2(math.sin(yaw-self.checkpoint_pose[2]), math.cos(yaw-self.checkpoint_pose[2])))
                if (not all(math.isfinite(v) for v in [*velocity, *xy, yaw])
                        or max(abs(v) for v in velocity) >= .003
                        or math.dist(xy, self.checkpoint_pose[:2]) > .005 or delta_yaw > .005):
                    self.checkpoint_motion = True
            if self.last_xy is not None:
                step = math.dist(xy, self.last_xy)
                if step > .25:
                    self.external_command = 'odometry_jump'
                self.distance += step
            self.last_xy = xy
            if not self.trajectory or time.time()-self.trajectory[-1][0] > .2:
                self.trajectory.append([time.time(), *xy, angle(msg.pose.pose.orientation),
                                        msg.twist.twist.linear.x, msg.twist.twist.angular.z])
        elif key == 'explore':
            self.event('explore_status', status=msg.status)
        elif key == 'graph':
            g = graph_summary(msg)
            stamp = time.time()
            if not self.graphs or (g['node_count'], g['edge_count']) != (self.graphs[-1]['node_count'], self.graphs[-1]['edge_count']):
                self.graph_counts.append(dict(time=stamp, node_count=g['node_count'], edge_count=g['edge_count']))
            # Reproject anchored semantic observations through current native poses,
            # including optimization updates that preserve the node/edge counts.
            self.graphs[:] = [dict(time=stamp, **g)]

    def native_command(self, msg):
        with self.feedback_lock:
            self.native_commands.append((time.monotonic(), [msg.linear.x, msg.linear.y, msg.angular.z]))

    def command(self, msg, info):
        if not self.moving_session:
            return
        if max(abs(msg.linear.x), abs(msg.linear.y), abs(msg.angular.z)) <= .0001:
            return
        # Installed Jazzy rclpy supplies timestamp metadata but no publisher GID.
        # Check actual hardware commands against fresh native upstream commands;
        # do not claim publisher attribution that this runtime cannot provide.
        with self.feedback_lock:
            self.pending_commands.append((time.monotonic(), [msg.linear.x, msg.linear.y, msg.angular.z]))

    def event(self, name, **data):
        row = dict(time=time.time(), phase=self.phase, event=name, **data)
        with self.feedback_lock:
            self.events.append(row)
            with (self.evidence/'events.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')

    def pump(self, seconds, guarded=False):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            time.sleep(min(.01, max(0., end-time.monotonic())))
            # Check at 40 Hz, rather than repeating ROS graph / body decoding
            # after every individual callback and starving incoming telemetry.
            # Native Collision Monitor remains independent and unchanged.
            if guarded and time.monotonic()-self.last_health_check >= .025:
                self.check()
                self.last_health_check = time.monotonic()

    def transform(self, source):
        t = self.buf.lookup_transform('map', source, rclpy.time.Time())
        age = (self.node.get_clock().now()-rclpy.time.Time.from_msg(t.header.stamp)).nanoseconds/1e9
        if not -.3 <= age < .65:
            raise RuntimeError('stale_map_transform')
        return t

    def check(self):
        if self.receiver_error or not self.receiver_thread.is_alive():
            raise RuntimeError('feedback_executor_stopped:' + str(self.receiver_error))
        if self.interrupted:
            raise RuntimeError('interrupted')
        if self.external_command:
            raise RuntimeError(self.external_command)
        now = time.monotonic()
        with self.feedback_lock:
            state, times = self.state.copy(), self.times.copy()
            now = time.monotonic()
            while self.pending_commands and now-self.pending_commands[0][0] > .05:
                stamp, command = self.pending_commands.popleft()
                candidates = [v for t,v in self.native_commands if -.05 <= stamp-t <= .3]
                if not command_matches_native(command, candidates):
                    raise RuntimeError('hardware_command_not_matched_to_native_navigation')
        if now-times.get('joints', 0) > .5:
            self.event('telemetry_guard_failed', reason='stale_body_posture',
                       arrival_ages_s={k: now-v for k,v in times.items()},
                       maximum_feedback_gap_s=dict(self.feedback_gaps))
            raise RuntimeError('stale_body_posture')
        joints = json.loads(state['joints'].data)
        # The calibrated footprint is valid for the measured native standing
        # arm pose. Stop if another controller changes that physical envelope.
        for group in ('left_arm', 'right_arm', 'leg_waist'):
            expected = self.body_reference[group+'_joint_state']['position']
            body_state = joints[group+'_joint_state']
            if any(body_state['communication_lost']) or len(body_state['position']) != len(expected):
                raise RuntimeError('body_feedback_invalid')
            if any(not math.isfinite(a) or abs(a-b) > 2 for a,b in zip(body_state['position'], expected)):
                raise RuntimeError('body_pose_outside_calibrated_footprint')
        if now-times.get('footprint', 0) > 2:
            raise RuntimeError('stale_collision_approach_footprint')
        if self.deadline is not None and now > self.deadline:
            raise RuntimeError('session_time_budget_reached')
        ages = {}
        for key in ['front', 'rear', 'merged', 'odom', 'battery']:
            m = state.get(key)
            header = (self.node.get_clock().now()-rclpy.time.Time.from_msg(m.header.stamp)).nanoseconds/1e9 if m else float('inf')
            ages[key] = [now-times.get(key, 0), header]
        odom = state.get('odom')
        velocity = [odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z] if odom else [0, 0, 0]
        battery = state.get('battery')
        try:
            fail_if_unhealthy(ages, velocity, battery.current if battery else 99, battery.power_supply_status if battery else 1,
                              self.distance, math.dist(self.last_xy, self.origin) if self.origin else 0)
        except RuntimeError as exc:
            self.event('telemetry_guard_failed', reason=str(exc), ages_s=ages, odom_velocity=velocity)
            raise
        if not math.isfinite(battery.percentage) or battery.percentage < 25:  # Native driver uses 0..100.
            raise RuntimeError('battery_below_25_percent')
        if now-times.get('map', 0) > 6:
            raise RuntimeError('stale_map')
        t = self.transform('odom')
        pose = [t.transform.translation.x, t.transform.translation.y, angle(t.transform.rotation)]
        if self.map_odom is not None:
            da = abs(math.atan2(math.sin(pose[2]-self.map_odom[2]), math.cos(pose[2]-self.map_odom[2])))
            if math.dist(pose[:2], self.map_odom[:2]) > .6 or da > .3:
                raise RuntimeError('localization_correction_requires_review')
        self.map_odom = pose
        if now-self.last_graph_check > 1:
            self.last_graph_check = now
            pubs = self.node.get_publishers_info_by_topic('/map')
            owners = [list(p.endpoint_gid) for p in pubs]
            if self.owner is not None and owners != self.owner:
                raise RuntimeError('map_publisher_changed')
            if len(owners) != 1:
                raise RuntimeError('map_publisher_count')
            self.owner = owners
            counts = collections.Counter(name for name, _ in self.node.get_node_names_and_namespaces())
            for name in ['controller_server', 'planner_server', 'bt_navigator', 'collision_monitor']:
                if counts[name] != 1:
                    raise RuntimeError('missing_or_duplicate_' + name)

    def wait(self, future, timeout, guarded=True):
        deadline = time.monotonic()+timeout
        while not future.done() and time.monotonic() < deadline:
            self.pump(.025, guarded)
        if not future.done():
            raise RuntimeError('native_api_timeout')
        return future.result()

    def parked(self):
        with self.feedback_lock:
            feedback = tuple(self.odom_feedback)
        return stationary_feedback_reason(feedback, time.monotonic(),
                                          self.node.get_clock().now().nanoseconds/1e9) == 'ok'

    def reacquire_stationary_feedback(self, timeout=5.):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            self.pump(.05)
            if self.checkpoint_motion:
                raise RuntimeError('robot_moved_during_stationary_checkpoint')
            if self.parked():
                return
        with self.feedback_lock:
            feedback = tuple(self.odom_feedback)
        reason = stationary_feedback_reason(feedback, time.monotonic(),
                                            self.node.get_clock().now().nanoseconds/1e9)
        raise RuntimeError('stationary_feedback_timeout:' + reason)

    def stop_explorer(self):
        # Wait for the upstream pause/cancel to finish before issuing a revisit.
        started = time.monotonic()
        deadline = started+10
        cancel_requested = False
        quiet_since = None
        self.pause.publish(Bool(data=False))
        while time.monotonic() < deadline:
            self.pump(.15)
            active = any(g.status in [1, 2, 3] for g in getattr(self.state.get('nav'), 'status_list', []))
            status = getattr(self.state.get('explore'), 'status', '')
            acknowledged = status == 'exploration_paused' and self.times.get('explore', 0) >= started
            if acknowledged and not cancel_requested:
                if not self.cancel_nav.wait_for_service(timeout_sec=1):
                    raise RuntimeError('native_cancel_service_unavailable')
                response = self.wait(self.cancel_nav.call_async(CancelGoal.Request()), 3, guarded=False)
                if response.return_code not in [0, 3]:
                    raise RuntimeError('native_cancel_rejected')
                cancel_requested = True
            if acknowledged and cancel_requested and not active and self.parked():
                quiet_since = quiet_since or time.monotonic()
                if time.monotonic()-quiet_since > 1:
                    return
            else:
                quiet_since = None
        raise RuntimeError('upstream_pause_not_confirmed')

    def spawn(self, command, log_name):
        stream = (self.folder/'logs'/log_name).open('w')
        self.children_logs.append(stream)
        return subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)

    def capture(self):
        if self.semantic is None or self.semantic.poll() is not None:
            self.semantic = self.spawn(['/bin/bash', str(self.folder/'capture-semantic-start.sh')], f'semantic-{int(time.time())}.log')

    def checkpoint(self, name):
        self.reacquire_stationary_feedback()
        with self.feedback_lock:
            odom = self.state['odom']
            self.checkpoint_pose = [odom.pose.pose.position.x, odom.pose.pose.position.y,
                                    angle(odom.pose.pose.orientation)]
            self.checkpoint_motion = False
        try:
            self.save_checkpoint(name)
            # Discard callback backlog before proving a fresh stationary span.
            with self.feedback_lock:
                self.odom_feedback.clear()
            self.reacquire_stationary_feedback()
            self.event('stationary_checkpoint_verified', checkpoint=name,
                       full_rate_odom_samples=len(self.odom_feedback))
        finally:
            with self.feedback_lock:
                self.checkpoint_pose = None
                self.checkpoint_motion = False

    def save_checkpoint(self, name):
        self.phase = 'save_' + name
        filename = str(self.maps/name)
        # SLAM's SaveMap callback shells out with its inherited DDS index cap.
        # Run that same upstream saver with this session's DDS configuration.
        saver = self.spawn(['/opt/ros/jazzy/lib/nav2_map_server/map_saver_cli', '-f', filename,
                            '--ros-args', '-p', 'save_map_timeout:=15.0',
                            '-p', 'map_subscribe_transient_local:=true'], self.evidence.name+'-'+name+'-map-save.log')
        deadline = time.monotonic()+20
        while saver.poll() is None and time.monotonic()<deadline:
            self.pump(.05)
        if saver.poll() is None:
            saver.terminate()
            raise RuntimeError('native_map_saver_timeout')
        if saver.returncode != 0:
            raise RuntimeError('native_map_saver_failed_'+str(saver.returncode))
        req = SerializePoseGraph.Request(filename=filename)
        if not self.serializer.wait_for_service(timeout_sec=2):
            raise RuntimeError('map_serialize_service_unavailable')
        result = self.wait(self.serializer.call_async(req), 20, guarded=False)
        if result.result != 0:
            raise RuntimeError('map_serialize_failed_'+str(result.result))
        with self.feedback_lock:
            graphs = list(self.graphs)
            graph_counts = list(self.graph_counts)
        (self.evidence/(name+'-graphs.json')).write_text(json.dumps(graphs, indent=2))
        (self.evidence/(name+'-graph-counts.json')).write_text(json.dumps(graph_counts, indent=2))
        self.event('checkpoint_saved', checkpoint=name, map_prefix=filename)

    def revisit(self, name):
        self.phase = name
        goal = NavigateToPose.Goal()
        goal.behavior_tree = str(self.folder/'config/native-navigation-recovery.xml')
        goal.pose = copy.deepcopy(self.anchor)
        goal.pose.header.stamp = self.node.get_clock().now().to_msg()
        self.return_handle = self.wait(self.navigate.send_goal_async(goal), 5)
        if not self.return_handle.accepted:
            raise RuntimeError('revisit_rejected')
        result = self.wait(self.return_handle.get_result_async(), 240)
        self.return_handle = None
        if result.status != 4:
            raise RuntimeError('revisit_failed_' + str(result.status))
        self.pump(3, guarded=True)
        actual = self.transform('base_link').transform.translation
        error = math.hypot(actual.x-self.anchor.pose.position.x, actual.y-self.anchor.pose.position.y)
        self.event('native_revisit_succeeded', position_error_m=error, loop_closure_verified=False)
        if error > .4 or not self.parked():
            raise RuntimeError('revisit_position_or_stop_not_verified')

    def initial_scan(self):
        self.phase = 'initial_scan'
        if not self.spin_action.wait_for_server(timeout_sec=3):
            raise RuntimeError('native_spin_unavailable')
        goal = Spin.Goal()
        goal.target_yaw = 2*math.pi
        goal.time_allowance.sec = 160
        self.event('native_initial_scan_started')
        self.return_handle = self.wait(self.spin_action.send_goal_async(goal), 5)
        if not self.return_handle.accepted:
            raise RuntimeError('native_spin_rejected')
        result = self.wait(self.return_handle.get_result_async(), 165)
        self.return_handle = None
        self.pump(2, guarded=True)
        self.event('native_initial_scan_finished', status=result.status)
        if result.status != 4 or not self.parked():
            raise RuntimeError('native_initial_scan_failed')
        self.checkpoint('initial-scan')

    def run(self, seconds, known_anchor=None):
        if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', NAV_UNIT]).returncode:
            raise RuntimeError('owned_navigation_unit_not_active')
        self.pump(4)
        self.check()
        if not self.parked() or any(g.status in [1, 2, 3] for g in getattr(self.state.get('nav'), 'status_list', [])):
            raise RuntimeError('robot_not_idle')
        if not self.navigate.wait_for_server(timeout_sec=2):
            raise RuntimeError('navigation_unavailable')
        t = self.transform('base_link')
        self.anchor = PoseStamped()
        self.anchor.header.frame_id = 'map'
        self.anchor.pose.position.x = t.transform.translation.x
        self.anchor.pose.position.y = t.transform.translation.y
        self.anchor.pose.orientation = t.transform.rotation
        if known_anchor is not None:
            saved=json.loads(known_anchor.read_text())
            if saved['frame'] != 'map' or self.owner != [saved['map_owner_gid']]:
                raise RuntimeError('known_anchor_map_epoch_changed')
            self.anchor.pose.position.x,self.anchor.pose.position.y=saved['position']
            self.anchor.pose.orientation.x=self.anchor.pose.orientation.y=0.
            self.anchor.pose.orientation.z=math.sin(saved['yaw']/2)
            self.anchor.pose.orientation.w=math.cos(saved['yaw']/2)
        (self.evidence/'anchor.json').write_text(json.dumps({'frame':'map','position':[self.anchor.pose.position.x,self.anchor.pose.position.y],'yaw':angle(self.anchor.pose.orientation),'map_owner_gid':self.owner[0]},indent=2))
        with self.feedback_lock:
            self.origin = self.last_xy[:]
            self.distance = 0.
            self.moving_session = True
        self.bag = self.spawn(['ros2', 'bag', 'record', '-s', 'mcap', '-o', str(self.folder/'bags'/self.evidence.name),
                              '/tf', '/tf_static', '/map', '/zeng306/mapping_scan', '/topic_gv_front_lidar_0_306', '/topic_gv_rear_lidar_0_306',
                              '/topic_gv_wheel_odom_0_306', '/topic_gv_target_cmd_vel_0_306', '/slam_toolbox/graph_visualization', '/explore/status'],
                             self.evidence.name+'-bag.log')
        self.checkpoint('coverage-start')
        started = time.monotonic()
        self.deadline = started+seconds
        last_capture = 0.
        if known_anchor is not None:
            self.revisit('continuity_revisit')
            self.checkpoint('continuity-revisit')
        if self.graphs and self.graphs[-1]['node_count'] < 50:
            self.initial_scan()
        for cycle in range(4):
            if time.monotonic()-started > seconds:
                break
            self.phase = 'frontier_' + str(cycle)
            self.check()
            config = 'explore.refine.yaml' if cycle >= 2 else 'explore.306.yaml'
            if self.native is None:
                self.native = self.spawn([EXPLORER, '--ros-args', '--params-file', str(self.folder/'config'/config)], f'explore-cycle-{cycle}.log')
            else:
                self.pause.publish(Bool(data=True))
            phase_start = time.monotonic()
            progress_time, progress_xy, progress_yaw = phase_start, self.last_xy[:], self.trajectory[-1][3]
            phase_reason = 'phase_budget'
            while time.monotonic()-phase_start < 240 and time.monotonic()-started < seconds:
                self.pump(.05, guarded=True)
                now = time.monotonic()
                if self.native.poll() is not None:
                    raise RuntimeError('native_explorer_exited')
                status = getattr(self.state.get('explore'), 'status', '')
                if status == 'exploration_complete' and self.times.get('explore', 0) > phase_start:
                    phase_reason = 'native_frontiers_exhausted'
                    break
                da = abs(math.atan2(math.sin(self.trajectory[-1][3]-progress_yaw), math.cos(self.trajectory[-1][3]-progress_yaw)))
                if math.dist(progress_xy, self.last_xy) > .12 or da > .4:
                    progress_time, progress_xy, progress_yaw = now, self.last_xy[:], self.trajectory[-1][3]
                if now-progress_time > 90:
                    phase_reason = 'no_physical_progress_90s'
                    break
                if now-last_capture > 90:
                    self.capture()
                    last_capture = now
                live = {'time': time.time(), 'phase': self.phase, 'distance_m': self.distance,
                        'pose_odom': self.trajectory[-1][1:4], 'explore_status': status,
                        'graph_nodes': len(self.graphs[-1]['nodes']) if self.graphs else None}
                if int(now*2) != getattr(self, '_last_write', None):
                    (self.evidence/'live.json').write_text(json.dumps(live, indent=2))
                    self._last_write = int(now*2)
            self.event('phase_finished', reason=phase_reason)
            self.stop_explorer()
            self.revisit('revisit_' + str(cycle))
            self.checkpoint('cycle-' + str(cycle))
            if cycle == 1:
                self.native.send_signal(signal.SIGINT)
                self.native.wait(timeout=5)
                self.native = None
            if cycle >= 2 and phase_reason == 'native_frontiers_exhausted':
                return 'native_frontiers_exhausted_requires_coverage_review'
        return 'monitored_segment_finished'

    def close(self, reason):
        if self.moving_session:
            self.pause.publish(Bool(data=False))
            if self.return_handle:
                try:
                    self.wait(self.return_handle.cancel_goal_async(), 3, guarded=False)
                except Exception:
                    pass
            if self.native and self.native.poll() is None:
                self.native.send_signal(signal.SIGINT)
            # End all owned controllers before the final zero; no competing goal survives.
            try:
                stopped = subprocess.run(['systemctl', '--user', 'stop', NAV_UNIT], timeout=15)
                if stopped.returncode:
                    self.event('owned_navigation_stop_failed', returncode=stopped.returncode)
            except subprocess.TimeoutExpired:
                subprocess.run(['systemctl', '--user', 'kill', '--signal=SIGKILL', NAV_UNIT], timeout=5)
                self.event('owned_navigation_force_stopped')
            finally:
                for _ in range(30):
                    self.zero.publish(Twist())
                    self.pump(.1)
            try:
                self.checkpoint('coverage-final')
            except Exception as exc:
                self.event('final_checkpoint_failed', error=str(exc))
        for child in [self.native, self.bag]:
            if child and child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.terminate()
        if self.semantic and self.semantic.poll() is None:
            # A standalone capture may finish after motion has safely stopped.
            self.event('semantic_capture_finishing', pid=self.semantic.pid)
        logs = subprocess.run(['journalctl', '--user', '-u', 'zeng306-upstream-slam', '--since', '@'+str(int(self.started_wall)), '--no-pager', '-o', 'cat'], capture_output=True, text=True, timeout=8).stdout
        (self.evidence/'slam.log').write_text(logs)
        closures = [line for line in logs.splitlines() if 'Loop closed!' in line]
        self.pump(1.4)
        parked_verified = self.parked()
        self.receiver.shutdown(timeout_sec=3.)
        self.receiver_thread.join(timeout=3.)
        report = {'reason': reason, 'motion_session_started': self.moving_session, 'parked_verified': self.parked(),
                  'distance_m': self.distance, 'events': self.events, 'trajectory': self.trajectory,
                  'loop_closure_log_events': closures, 'loop_closure_verified': bool(closures),
                  'environment_coverage_fraction': None, 'complete_environment_scan_verified': False}
        report['parked_verified'] = parked_verified
        report['maximum_feedback_gap_s'] = self.feedback_gaps
        report['feedback_executor_error'] = self.receiver_error
        (self.evidence/'result.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({k:v for k,v in report.items() if k not in ['events', 'trajectory']}))
        for stream in self.children_logs:
            stream.close()
        self.node.destroy_node()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('session', type=Path)
    p.add_argument('--run', action='store_true')
    p.add_argument('--strategy', choices=['reward', 'legacy'], default='reward')
    p.add_argument('--revisit-known-anchor', type=Path)
    p.add_argument('--seconds', type=float, default=1800)
    args = p.parse_args()
    if not args.run:
        from session_health import main as health
        import sys
        sys.argv = [sys.argv[0], str(args.session/'evidence'/'coverage-preflight.json')]
        return health()
    if not 60 <= args.seconds <= 1800:
        p.error('--seconds must be between 60 and 1800')
    rclpy.init()
    session = Session(args.session, 'coverage-' + time.strftime('%Y%m%d-%H%M%S'))
    def interrupt(*_):
        session.interrupted = True
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    reason = 'not_started'
    try:
        if args.strategy == 'reward':
            from exploration306.reward_runner import run_reward_session
            reason = run_reward_session(session, args.seconds)
        else:
            reason = session.run(args.seconds, args.revisit_known_anchor)
    except BaseException as exc:
        reason = 'error:' + str(exc)
    finally:
        try:
            session.close(reason)
        finally:
            rclpy.shutdown()
    raise SystemExit(1 if reason.startswith('error:') else 0)


if __name__ == '__main__':
    main()
