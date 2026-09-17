"""Isolated orchestration contract tests; fake navigation, never hardware control."""
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch

from nav_msgs.msg import OccupancyGrid
import rclpy

from exploration306.reward_policy import write_json
from exploration306.reward_runner import run_reward_session


class Done:
    def __init__(self, value): self.value=value
    def done(self): return True
    def result(self): return self.value


class FakeNative:
    candidates=[{'kind':'frontier','xy':[2.,2.],'yaw':0.,'visible_frontier_m':2.,'dp_rank':1}]
    def __init__(self, node, wait): pass
    def snapshot(self): return {'pose':[0.,0.,0.],'candidates':[c.copy() for c in self.candidates],'native_frontier_count':len(self.candidates)}
    def validate(self, candidate): return {**candidate,'path_valid':True,'path_length_m':2.5}


class RunnerContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert os.environ.get('ROS_DOMAIN_ID')=='224'
        rclpy.init()

    @classmethod
    def tearDownClass(cls): rclpy.shutdown()

    def run_case(self, status=4, empty=False, vision_valid=True, reconnect=False):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); evidence=folder/'evidence'; evidence.mkdir()
            write_json(folder/'config/reward.306.json', {'maximum_goals':3 if reconnect else 1,'maximum_empty_rounds':1})
            (folder/'semantics').mkdir()
            node=rclpy.create_node('isolated_reward_runner_contract')
            msg=OccupancyGrid();msg.info.width=10;msg.info.height=10;msg.info.resolution=.1
            msg.info.origin.orientation.w=1.;msg.data=[0]*50+[-1]*50
            s=N(folder=folder,evidence=evidence,node=node,owner=[[1]*16],state={'map':msg},
                last_xy=[0.,0.],distance=0.,graphs=[{'nodes':{},'edges':[]}],feedback_lock=threading.RLock(),
                return_handle=None,semantic=None,deadline=None,events=[],goals=[])
            s.pump=lambda *a,**k:None; s.check=lambda:None; s.parked=lambda:True
            s.reacquire_stationary_feedback=lambda:None
            s.wait=lambda future,*a,**k:future.result()
            s.spawn=lambda *a:N(poll=lambda:None)
            s.checkpoint=lambda name:s.events.append(('checkpoint',name))
            s.event=lambda event,**data:s.events.append((event,data))
            s.transform=lambda source:N(transform=N(translation=N(x=2.,y=2.)))
            def send(goal):
                s.goals.append(goal)
                def result():
                    s.distance=2.5
                    msg.data=[0]*70+[-1]*30
                    return Done(N(status=status,result=N(error_code=0 if status==4 else 208)))
                return Done(N(accepted=True,get_result_async=result))
            s.navigate=N(wait_for_server=lambda **k:True,send_goal_async=send)
            def capture():
                write_json(folder/'semantics/fake.json',{'schema_valid':vision_valid,
                    'semantic_scene':{'view_quality':'good','place_features':['glass_partition']}})
                s.semantic=N(poll=lambda:0 if vision_valid else 2)
            s.capture=capture
            fake=type('PerCaseNative',(FakeNative,),{'candidates':[] if empty else FakeNative.candidates})
            try:
                with patch('exploration306.reward_runner.subprocess.run',return_value=N(returncode=0)), \
                     patch('exploration306.reward_runner.NativeCandidates',fake), \
                     patch('exploration306.reward_runner.history_recovery_candidates',return_value=
                           [{'kind':'revisit_recovery','xy':[2.,2.],'yaw':0.,'revisit_value':1.}] if reconnect else []):
                    if status != 4:
                        with self.assertRaisesRegex(RuntimeError,'reward_native_navigation_failed'):
                            run_reward_session(s,120)
                    else:
                        run_reward_session(s,120)
                memory=folder/'reward/experience.json'
                rows=json.loads(memory.read_text())['episodes'] if memory.exists() else []
                self.assertEqual(len(s.goals),0 if empty and not reconnect else 1)
                if not empty or reconnect:
                    self.assertTrue(s.goals[0].behavior_tree.endswith('native-navigation-reward.xml'))
                    self.assertEqual(s.goals[0].pose.header.frame_id,'map')
                    self.assertEqual(rows[-1]['success'],status==4)
                    if status==4:
                        self.assertAlmostEqual(rows[-1]['new_known_m2'],.2,places=6)
                        self.assertEqual(rows[-1]['new_visual_features'],1 if vision_valid else 0)
                    else:
                        self.assertEqual(rows[-1]['reward_terms']['failure'],-3)
                self.assertFalse(node.get_publishers_info_by_topic('/topic_gv_target_cmd_vel_0_306'))
            finally: node.destroy_node()

    def test_success_records_real_evidence_contract(self): self.run_case()
    def test_navigation_failure_is_remembered_without_another_goal(self): self.run_case(status=6)
    def test_no_reachable_candidates_sends_no_navigation_goal(self): self.run_case(empty=True)
    def test_vision_failure_preserves_geometry_based_progress(self): self.run_case(vision_valid=False)
    def test_restart_history_reconnect_is_limited_to_one_navigation_goal(self): self.run_case(empty=True,reconnect=True)


if __name__=='__main__':unittest.main(verbosity=2)
