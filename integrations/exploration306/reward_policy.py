"""Reward/experience glue; no frontier search, path planner, SLAM or actuator code.

Expected frontier visibility comes from the upstream C++ core. Executed rewards
use observed map changes and uniquely identified native graph constraints. These
are exploration heuristics, not trained RL or proof of global loop closure.
"""
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np


DEFAULTS = {
    'information_weight': 4.0, 'native_dp_weight': 1.0,
    'revisit_weight': 1.5, 'semantic_weight': 0.5,
    'travel_penalty_per_m': 0.08, 'repeat_penalty': 0.6,
    'failure_radius_m': 1.0, 'failure_cooldown_s': 600.0,
    'revisit_min_travel_m': 8.0, 'revisit_max_distance_m': 8.0,
    'revisit_min_distance_m': 1.2, 'revisit_candidate_separation_m': 2.0,
    'revisit_candidate_limit': 4, 'history_recovery_max_per_run': 1,
    'history_recovery_max_path_m': 3.0,
    'revisit_min_node_gap': 50, 'revisit_min_graph_path_m': 8.0,
    'minimum_new_area_m2': 0.05, 'maximum_goal_path_m': 15.0,
    'minimum_goal_distance_m': 0.65, 'maximum_goals': 30,
    'graph_first_live_node': 188, 'maximum_empty_rounds': 3,
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


class MapGain:
    """Union on the map's world lattice; resizing and repeated cells earn zero.

    A change of resolution, rotation or fractional lattice origin resets the
    baseline. Map corrections can still resemble new evidence: never label this
    as independently measured area or use it as collision clearance.
    """
    def __init__(self):
        self.origin = self.resolution = self.known = self.epoch = None

    def update(self, data, resolution, origin, epoch, yaw=0.0):
        data = np.asarray(data, dtype=np.int8)
        if data.ndim != 2 or not data.size or resolution <= 0 or not all(map(math.isfinite, [resolution, *origin, yaw])):
            raise ValueError('invalid_occupancy_grid')
        reset = self.known is None or epoch != self.epoch or abs(yaw) > 1e-7 or abs(resolution-(self.resolution or resolution)) > 1e-8
        offset = np.zeros(2, dtype=int)
        if not reset:
            exact = (np.asarray(origin)-self.origin)/resolution
            offset = np.rint(exact).astype(int)
            reset = bool(np.max(np.abs(exact-offset)) > 0.01)
        if reset:
            self.origin, self.resolution, self.epoch = np.array(origin, dtype=float), resolution, epoch
            self.known = data >= 0
            return {'new_known_m2': 0.0, 'new_free_m2': 0.0, 'baseline_reset': True}
        old_h, old_w = self.known.shape
        h, w = data.shape
        left, bottom = min(0, offset[0]), min(0, offset[1])
        right, top = max(old_w, offset[0]+w), max(old_h, offset[1]+h)
        if (right-left)*(top-bottom) > 16_000_000:
            raise ValueError('map_union_geometry_exceeds_session_bounds')
        merged = np.zeros((top-bottom, right-left), dtype=bool)
        merged[-bottom:old_h-bottom, -left:old_w-left] = self.known
        x, y = offset-[left, bottom]
        previous = merged[y:y+h, x:x+w]
        new = (data >= 0) & ~previous
        result = {'new_known_m2': float(np.count_nonzero(new)*resolution**2),
                  'new_free_m2': float(np.count_nonzero(new & (data == 0))*resolution**2),
                  'baseline_reset': False}
        previous |= data >= 0
        self.known = merged
        self.origin += np.array([left, bottom])*resolution
        return result


def graph_geometry(graph, first_node=188):
    nodes = {int(k): tuple(v[:2]) for k, v in graph.get('nodes', {}).items()
             if int(k) >= first_node and len(v) >= 2 and all(map(math.isfinite, v[:2]))}
    prefix, distance, previous = {}, 0.0, None
    for node_id in sorted(nodes):
        if previous is not None:
            distance += math.dist(nodes[node_id], nodes[previous])
        prefix[node_id], previous = distance, node_id
    return nodes, prefix


def cross_visit_pairs(graph, first_node=188):
    """Read native edges; qualify nonlocal revisits, not dedicated closure events."""
    nodes, prefix = graph_geometry(graph, first_node)
    pairs = set()
    for edge in graph.get('edges', []):
        ids = edge.get('node_ids_from_positions', [])
        if len(ids) != 2 or any(i is None for i in ids) or edge.get('localization_edge'):
            continue
        a, b = sorted(map(int, ids))
        if a in nodes and b in nodes and b-a >= 50 and prefix[b]-prefix[a] >= 5.0 and math.dist(nodes[a], nodes[b]) <= 1.0:
            pairs.add((a, b))
    return pairs


class GraphGain:
    def __init__(self):
        self.seen = set()
        self.epoch = None

    def update(self, graph, epoch):
        pairs = cross_visit_pairs(graph)
        if self.epoch != epoch:
            self.epoch, self.seen = epoch, pairs
            return 0
        count = len(pairs-self.seen)
        self.seen |= pairs
        return count


class Experience:
    def __init__(self, epoch, path=None, config=None):
        self.epoch, self.path = epoch, Path(path) if path else None
        self.config = {**DEFAULTS, **(config or {})}
        self.rows = []
        if self.path and self.path.exists():
            saved = json.loads(self.path.read_text(encoding='utf-8'))
            if saved.get('map_epoch') == epoch:
                self.rows = saved['episodes'][-300:]

    def nearby(self, xy):
        return [r for r in self.rows if math.dist(xy, r['xy']) < self.config['failure_radius_m']]

    def blocked(self, xy, now):
        return any(not r['success'] and 0 <= now-r['time'] < self.config['failure_cooldown_s'] for r in self.nearby(xy))

    def record(self, row):
        self.rows = (self.rows+[row])[-300:]
        if self.path:
            write_json(self.path, {'schema': 1, 'map_epoch': self.epoch, 'episodes': self.rows})

    def score(self, candidate, now):
        c, p = candidate, self.config
        if not c.get('path_valid') or self.blocked(c['xy'], now):
            return None
        length = c.get('path_length_m', float('inf'))
        limit = min(p['maximum_goal_path_m'], p['history_recovery_max_path_m']) if c['kind'] == 'revisit_recovery' else p['maximum_goal_path_m']
        if not math.isfinite(length) or not p['minimum_goal_distance_m'] <= length <= limit:
            return None
        for key in ('visible_frontier_m', 'revisit_value', 'semantic_novelty'):
            if not math.isfinite(c.get(key, 0)):
                return None
        repeats = sum(1 for r in self.nearby(c['xy']) if r['success'] and r.get('new_known_m2', 0) < p['minimum_new_area_m2'])
        terms = {
            'information': p['information_weight']*min(1.0, max(0.0, c.get('visible_frontier_m', 0))/2.0),
            'native_dp': p['native_dp_weight']/max(1, c.get('dp_rank', 1)) if c['kind'] == 'frontier' else 0.0,
            'revisit': p['revisit_weight']*min(1.0, max(0.0, c.get('revisit_value', 0))),
            'semantic': p['semantic_weight']*min(1.0, max(0.0, c.get('semantic_novelty', 0))),
            'travel': -p['travel_penalty_per_m']*length,
            'repeat': -p['repeat_penalty']*min(5, repeats),
        }
        return {**c, 'score': sum(terms.values()), 'score_terms': terms}

    def rank(self, candidates, now):
        scored = [self.score(c, now) for c in candidates]
        return sorted([s for s in scored if s is not None], key=lambda c: (-c['score'], c['path_length_m']))


def revisit_candidates(graph, pose, travel_since_revisit, config=None):
    p = {**DEFAULTS, **(config or {})}
    if travel_since_revisit < p['revisit_min_travel_m']:
        return []
    nodes, prefix = graph_geometry(graph, p['graph_first_live_node'])
    if not nodes:
        return []
    latest = max(nodes)
    degrees = Counter(i for edge in cross_visit_pairs(graph, p['graph_first_live_node']) for i in edge)
    options = []
    for i, xy in nodes.items():
        gap = prefix[latest]-prefix[i]
        distance = math.dist(pose[:2], xy)
        if latest-i < p['revisit_min_node_gap'] or gap < p['revisit_min_graph_path_m'] or not p['revisit_min_distance_m'] < distance < p['revisit_max_distance_m']:
            continue
        value = min(1.0, gap/20)/(1+degrees[i]/5)
        options.append({'kind': 'revisit', 'node_id': i, 'xy': list(xy), 'yaw': pose[2],
                        'revisit_value': value, 'graph_path_gap_m': gap, 'existing_cross_visit_degree': degrees[i]})
    selected = []
    for c in sorted(options, key=lambda x: -x['revisit_value']):
        if all(math.dist(c['xy'], other['xy']) >= p['revisit_candidate_separation_m'] for other in selected):
            selected.append(c)
        if len(selected) == p['revisit_candidate_limit']:
            break
    return selected


def history_recovery_candidates(graph, pose, config=None):
    """One bounded reconnect opportunity after restart, using actual native history.

    This does not invent current-run mileage. The existing node/path separation
    requirements still apply, and every candidate needs fresh native planning.
    """
    p = {**(config or {}), 'revisit_min_travel_m': 0.0, 'revisit_min_distance_m': 0.8,
         'revisit_max_distance_m': 3.0, 'revisit_candidate_separation_m': 0.8,
         'revisit_candidate_limit': 12}
    return [{**c, 'kind': 'revisit_recovery', 'reason': 'frontiers_unreachable_reconnect_native_history'}
            for c in revisit_candidates(graph, pose, 0.0, p)]


def semantic_novelty(candidate, observations, graph, epoch):
    """Spatial novelty only from graph-anchored views in the current map epoch.

    Legacy unanchored labels remain available in reports, never treated as
    object coordinates or projected into unknown space.
    """
    nodes = graph.get('nodes', {})
    novelty = 1.0
    for obs in observations:
        anchor = obs.get('graph_anchor') or {}
        if not obs.get('schema_valid') or anchor.get('map_epoch') != epoch:
            continue
        xy = nodes.get(str(anchor.get('node_id')), nodes.get(anchor.get('node_id')))
        if xy is None or math.dist(candidate['xy'], xy[:2]) > 1.5:
            continue
        quality = obs.get('semantic_scene', {}).get('view_quality', 'limited')
        novelty = min(novelty, {'good': 0.0, 'limited': 0.5, 'poor': 0.8}.get(quality, 0.5))
    return novelty


def load_semantics(folder):
    records = []
    for file in Path(folder).glob('*.json'):
        try:
            record = json.loads(file.read_text(encoding='utf-8'))
            if record.get('schema_valid') and isinstance(record.get('semantic_scene'), dict):
                records.append(record)
        except (ValueError, OSError):
            continue
    return records


def observed_reward(new_area, new_constraints, new_visual_features, path_m, success):
    # Cap each component so edge density / verbose VLM labels cannot dominate.
    terms = {'map_evidence': 4*min(1.0, max(0.0, new_area)/2),
             'cross_visit_constraint': 1.5*min(1.0, max(0, new_constraints)/3),
             'visual_evidence': 0.5*min(1.0, max(0, new_visual_features)/3),
             'travel': -0.08*max(0, path_m), 'failure': 0.0 if success else -3.0}
    return {'reward': sum(terms.values()), 'reward_terms': terms,
            'loop_closure_verified': False, 'reward_type': 'bounded_observed_evidence_heuristic'}
