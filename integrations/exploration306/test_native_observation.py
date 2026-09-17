"""Observation-goal contracts in isolated DDS; no physical navigation client."""
import os
from types import SimpleNamespace as N
import unittest
import rclpy
from nav_msgs.msg import OccupancyGrid, Path
from exploration306.reward_runner import NativeCandidates, pose_msg

class Done:
    def __init__(self, value): self.value = value
    def result(self): return self.value

class ObservationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert os.environ.get('ROS_DOMAIN_ID') == '224'
        rclpy.init()

    @classmethod
    def tearDownClass(cls): rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('isolated_observation_goal_contract')
        self.native = NativeCandidates.__new__(NativeCandidates)
        self.native.node = self.node
        self.native.wait = lambda f, *args: f.result()
        self.native.robot_pose = [1., 2., 0.]
        grid = OccupancyGrid()
        grid.header.frame_id = 'map'
        grid.info.width = grid.info.height = 100
        grid.info.resolution = .1
        grid.info.origin.orientation.w = 1.
        grid.data = [0]*10000
        self.native.grid = grid
        self.requests = []
        self.footprint_valid = True
        self.candidate = {'kind': 'frontier', 'xy': [4., 2.], 'yaw': 0., 'visible_frontier_m': 1.}
        def plan(goal):
            xy = [goal.goal.pose.position.x, goal.goal.pose.position.y]
            self.requests.append(xy)
            path = Path()
            path.header.frame_id = 'map'
            if xy == self.candidate['xy']:
                return Done(N(accepted=True, get_result_async=lambda: Done(N(status=6, result=N(error_code=208, path=path)))))
            end = [3., 2.] if len(self.requests) == 2 else xy
            path.poses = [pose_msg(self.node, [1., 2.], 0.), pose_msg(self.node, [2.05, 2.05], 0.), pose_msg(self.node, end, 0.)]
            return Done(N(accepted=True, get_result_async=lambda: Done(N(status=4, result=N(error_code=0, path=path)))))
        self.native.planner = N(wait_for_server=lambda **k: True, send_goal_async=plan)
        self.native.validator = N(wait_for_service=lambda **k: True,
            call_async=lambda req: Done(N(is_valid=self.footprint_valid, invalid_pose_indices=[] if self.footprint_valid else [1])))

    def tearDown(self):
        self.assertFalse(self.node.get_publishers_info_by_topic('/topic_gv_target_cmd_vel_0_306'))
        self.node.destroy_node()

    def test_native_endpoint_becomes_explicit_replanned_observation_goal(self):
        result = self.native.validate(self.candidate)
        self.assertTrue(result['path_valid'])
        self.assertTrue(result['native_endpoint_replanned'])
        self.assertEqual(result['xy'], [3., 2.])
        self.assertEqual(self.requests[-1], result['xy'])
        self.assertEqual(result['source_frontier_xy'], [4., 2.])
        self.assertLessEqual(result['endpoint_error_m'], .15)

    def test_native_footprint_failure_is_never_accepted(self):
        self.footprint_valid = False
        self.assertFalse(self.native.validate(self.candidate)['path_valid'])

    def test_path_crossing_unknown_remains_rejected(self):
        self.native.grid.data[20*100+20] = -1
        self.assertFalse(self.native.validate(self.candidate)['path_valid'])

    def test_occupied_source_goal_does_not_trigger_observation_fallback(self):
        self.native.grid.data[19*100+39] = 100
        self.native.grid.data[20*100+40] = 100
        self.assertFalse(self.native.validate(self.candidate)['path_valid'])
        self.assertEqual(self.requests, [])

    def test_history_goal_is_not_offset(self):
        self.assertFalse(self.native.validate({**self.candidate, 'kind': 'revisit'})['path_valid'])
        self.assertEqual(len(self.requests), 1)

if __name__ == '__main__': unittest.main(verbosity=2)
