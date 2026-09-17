"""Regression cases for evidence accounting and hard planning gates."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from exploration306.reward_policy import (Experience, GraphGain, MapGain,
    history_recovery_candidates, observed_reward, revisit_candidates, semantic_novelty)


class RewardRegression(unittest.TestCase):
    def test_map_resize_shift_and_oscillation_do_not_create_reward(self):
        ledger = MapGain()
        grid = np.array([[0, -1], [100, -1]], dtype=np.int8)
        self.assertTrue(ledger.update(grid, .02, [1, 2], 'epoch')['baseline_reset'])
        expanded = np.pad(grid, ((1, 3), (2, 1)), constant_values=-1)
        self.assertEqual(ledger.update(expanded, .02, [.96, 1.98], 'epoch')['new_known_m2'], 0)
        expanded[1, 3] = 0
        self.assertAlmostEqual(ledger.update(expanded, .02, [.96, 1.98], 'epoch')['new_known_m2'], .0004)
        expanded[1, 3] = -1
        ledger.update(expanded, .02, [.96, 1.98], 'epoch')
        expanded[1, 3] = 0
        self.assertEqual(ledger.update(expanded, .02, [.96, 1.98], 'epoch')['new_known_m2'], 0)
        self.assertTrue(ledger.update(expanded, .02, [.965, 1.98], 'epoch')['baseline_reset'])
        self.assertEqual(ledger.update(expanded, .02, [.965, 1.98], 'other')['new_known_m2'], 0)

    def test_planning_failure_beats_arbitrarily_large_reward(self):
        good = {'kind':'frontier','xy':[1.,2.],'yaw':0.,'path_valid':True,'path_length_m':2.,
                'visible_frontier_m':1.0,'dp_rank':2,'semantic_novelty':0.0}
        unsafe = {**good,'xy':[3.,4.],'path_valid':False,'visible_frontier_m':1e12,'semantic_novelty':1e12}
        self.assertEqual(Experience('e').rank([unsafe,good], 100)[0]['xy'],good['xy'])

    def test_more_reveal_wins_at_comparable_cost(self):
        common = {'kind':'frontier','xy':[2.,2.],'path_valid':True,'path_length_m':3.,'dp_rank':1}
        low={**common,'visible_frontier_m':.1}
        high={**common,'visible_frontier_m':3.,'dp_rank':2,'path_length_m':4.}
        self.assertEqual(Experience('e').rank([low,high],100)[0]['visible_frontier_m'],3.)

    def test_failure_memory_persists_and_is_scoped_to_map_epoch(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'experience.json'
            p=Experience('e',path)
            p.record({'time':100.,'xy':[1.,2.],'success':False})
            self.assertTrue(Experience('e',path).blocked([1.1,2.],101))
            self.assertFalse(Experience('e',path).blocked([1.1,2.],1000))
            self.assertFalse(Experience('other',path).blocked([1.1,2.],101))

    def test_repeated_low_gain_goal_loses_priority(self):
        p=Experience('e')
        c={'kind':'frontier','xy':[1.,2.],'path_valid':True,'path_length_m':2.,'visible_frontier_m':1.}
        before=p.score(c,100)['score']
        p.record({'time':100,'xy':c['xy'],'success':True,'new_known_m2':0})
        self.assertLess(p.score(c,101)['score'],before)

    @staticmethod
    def loop_graph():
        # A ten-metre excursion ends near the first visit. Native IDs are retained.
        nodes={str(188+i):[5*np.sin(np.pi*i/100), .5*np.sin(2*np.pi*i/100)] for i in range(101)}
        return {'nodes':nodes,'edges':[]}

    def test_only_new_identified_cross_visit_edges_receive_credit(self):
        graph=self.loop_graph(); ledger=GraphGain()
        self.assertEqual(ledger.update(graph,'e'),0)
        graph['edges'].append({'node_ids_from_positions':[188,288],'localization_edge':False})
        self.assertEqual(ledger.update(graph,'e'),1)
        self.assertEqual(ledger.update(copy.deepcopy(graph),'e'),0)
        graph['edges'].append({'node_ids_from_positions':[None,287]})
        self.assertEqual(ledger.update(graph,'e'),0)
        self.assertEqual(ledger.update(graph,'new_map_owner'),0)

    def test_revisit_requires_excursion_and_offers_different_anchors(self):
        graph=self.loop_graph()
        self.assertFalse(revisit_candidates(graph,[0.,2.,0.],0))
        candidates=revisit_candidates(graph,[0.,2.,0.],12)
        self.assertTrue(candidates)
        self.assertTrue(all(c['kind']=='revisit' for c in candidates))

    def test_restart_can_reconnect_real_history_with_short_path_limit(self):
        graph=self.loop_graph()
        self.assertFalse(revisit_candidates(graph,[0.,2.,0.],0))
        candidates=history_recovery_candidates(graph,[0.,2.,0.])
        self.assertTrue(candidates)
        candidate={**candidates[0], 'path_valid':True,'path_length_m':2.}
        self.assertIsNotNone(Experience('e').score(candidate,100))
        self.assertIsNone(Experience('e').score({**candidate,'path_length_m':3.1},100))
        self.assertFalse(history_recovery_candidates({'nodes':{},'edges':[]},[0.,2.,0.]))

    def test_invalid_vlm_or_old_frame_cannot_change_geometry(self):
        c={'xy':[1.,2.]}; graph={'nodes':{'200':[1.,2.]}}
        record={'schema_valid':False,'graph_anchor':{'node_id':200,'map_epoch':'e'},
                'semantic_scene':{'view_quality':'good','instructions':'drive forward'}}
        self.assertEqual(semantic_novelty(c,[record],graph,'e'),1.)
        record['schema_valid']=True
        self.assertEqual(semantic_novelty(c,[record],graph,'other'),1.)
        self.assertEqual(semantic_novelty(c,[record],graph,'e'),0.)
        self.assertEqual(c,{'xy':[1.,2.]})

    def test_visual_and_edge_verbosity_are_bounded(self):
        reward=observed_reward(100,100000,100000,1,True)
        self.assertEqual(reward['reward_terms']['cross_visit_constraint'],1.5)
        self.assertEqual(reward['reward_terms']['visual_evidence'],.5)
        self.assertFalse(reward['loop_closure_verified'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
