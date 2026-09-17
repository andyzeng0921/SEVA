"""Boundary checks for robot supervision; no ROS node or motion publisher."""
import copy
import math
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from coverage_session import fail_if_unhealthy, Session


class Guards(unittest.TestCase):
    def setUp(self):
        self.args = dict(ages={k: [.1, .1] for k in ['front', 'rear', 'merged', 'odom', 'battery']},
                         velocity=[.1, 0., .08], battery_current=-4., battery_status=2, distance=20., radius=8.)

    def test_healthy(self):
        fail_if_unhealthy(**self.args)

    def test_one_missing_rear_stops(self):
        del self.args['ages']['rear']
        with self.assertRaisesRegex(RuntimeError, 'rear'):
            fail_if_unhealthy(**self.args)

    def test_replayed_lidar_and_future_clock_stop(self):
        for age in [2., -.5, math.nan, math.inf]:
            with self.subTest(age=age), self.assertRaisesRegex(RuntimeError, 'front'):
                self.args['ages']['front'] = [.01, age]
                fail_if_unhealthy(**self.args)

    def test_charging_and_invalid_battery_stop(self):
        for current, status in [(1.4, 1), (-1., 1), (0., 4), (math.nan, 2)]:
            with self.subTest(current=current, status=status), self.assertRaisesRegex(RuntimeError, 'undock'):
                fail_if_unhealthy(**dict(self.args, battery_current=current, battery_status=status))

    def test_overspeed_lateral_and_nan_stop(self):
        for velocity in [[.2, 0., 0.], [0., .03, 0.], [0., 0., .2], [math.nan, 0., 0.]]:
            with self.subTest(velocity=velocity), self.assertRaisesRegex(RuntimeError, 'speed'):
                fail_if_unhealthy(**dict(self.args, velocity=velocity))

    def test_distance_limit(self):
        with self.assertRaisesRegex(RuntimeError, 'distance'):
            fail_if_unhealthy(**dict(self.args, radius=26.))

    def test_stale_pause_cannot_authorize_revisit(self):
        s = Session.__new__(Session)
        now = [10.]
        s.state = {'explore': SimpleNamespace(status='exploration_paused')}
        s.times = {'explore': 9.}
        s.pause = SimpleNamespace(publish=lambda _: None)
        s.pump = lambda seconds: now.__setitem__(0, now[0]+seconds)
        s.parked = lambda: True
        with patch('coverage_session.time.monotonic', lambda: now[0]):
            with self.assertRaisesRegex(RuntimeError, 'pause_not_confirmed'):
                s.stop_explorer()

    def test_revisit_waits_for_cancel_and_quiet(self):
        s = Session.__new__(Session)
        now = [10.]
        s.state = {'explore': SimpleNamespace(status='exploration_paused'),
                   'nav': SimpleNamespace(status_list=[SimpleNamespace(status=2)])}
        s.times = {'explore': 10.01}
        s.pause = SimpleNamespace(publish=lambda _: None)
        s.parked = lambda: True
        def pump(seconds):
            now[0] += seconds
            if now[0] > 12:
                s.state['nav'].status_list = []
        s.pump = pump
        requests = []
        s.cancel_nav = SimpleNamespace(wait_for_service=lambda **_: True,
                                      call_async=lambda request: requests.append(request))
        s.wait = lambda *_, **kw: SimpleNamespace(return_code=0)
        with patch('coverage_session.time.monotonic', lambda: now[0]):
            s.stop_explorer()
        self.assertEqual(len(requests), 1)
        self.assertGreater(now[0], 13.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
