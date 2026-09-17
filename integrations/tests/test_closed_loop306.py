"""Failure injection for the adapter boundary; never publishes robot commands."""
import copy
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from zeng_agent.bootstrap import build_service
from zeng_agent.config import AgentConfig
from zeng_agent.models import ExecutionResult, AsrResult, CommandText
from upstream306.app import create_app
from upstream306.bridge import UpstreamBrain
from upstream306.contracts import normalize
from upstream306.observations import capture_rgbd
from upstream306.reflect_adapter import summarize
from upstream306.source import PROJECT


@pytest.fixture
def brain(tmp_path):
    config = AgentConfig.from_yaml(PROJECT / 'config.yaml')
    config.dry_run, config.llm.enabled, config.openclaw_robot.enabled = True, False, False
    return UpstreamBrain(build_service(config).brain_orchestrator, runtime=tmp_path, auto_summary=False)


@pytest.mark.parametrize('intent,slots,key,value', [
    ('navigate_to_waypoint', {'waypoint': 'home'}, 'name', 'home'),
    ('save_waypoint', {'name': 'point_1'}, 'name', 'point_1'),
    ('arm_preset', {'preset': 'reset'}, 'action_name', 'reset'),
    ('arm_continuous_move', {'direction': 'up', 'distance_m': .01}, 'amount_meters', .01),
])
def test_catalog_aliases_match_executor(brain, intent, slots, key, value):
    assert normalize(brain.catalog.by_intent(intent), slots)[key] == value


@pytest.mark.parametrize('slots', [
    {'direction': 'up', 'distance_m': .01, 'amount_meters': .1},
    {'direction': 'up', 'distance_m': float('nan')},
    {'direction': 'up', 'amount_meters': .51},
    {'direction': 'up', 'amount_meters': 0},
    {'direction': 'up', 'amount_meters': .01, 'limb': 'both'},
    {'direction': 'up', 'amount_meters': .01, 'unexpected': 1},
])
def test_bad_future_step_blocks_entire_plan(brain, slots):
    result = brain.run_steps([{'intent': 'robot_status'}, {'intent': 'arm_continuous_move', 'slots': slots}], confirmed=True)
    assert result['failure_code'] == 'INVALID_PARAMETERS'
    assert result['steps'][0]['controller_status'] == 'not_sent'
    assert all(e['kind'] != 'execution_started' for e in brain.store.events(result['task_id']))


def test_arm_reset_is_never_exposed_as_stop(brain):
    result = brain.run_steps([{'intent': 'arm_stop'}], confirmed=True)
    assert result['status'] == 'blocked'
    assert result['steps'][0]['failure_code'] == 'ARM_STOP_UNVERIFIED'
    assert result['steps'][0]['controller_status'] == 'not_sent'


def test_live_motion_remains_uncommissioned_even_with_confirmation(brain):
    brain.config.dry_run = False
    result = brain.run_steps([{'intent': 'arm_continuous_move', 'slots': {'direction': 'up', 'distance_m': .01}}], confirmed=True)
    assert result['steps'][0]['failure_code'] == 'HARDWARE_UNCOMMISSIONED'


def test_transient_read_recovery_continues_only_after_fresh_success(brain):
    class Backend:
        count = 0
        def robot_status(self):
            self.count += 1
            return ExecutionResult(self.count == 2, 'fixture', {'stdout': '/test_node' if self.count == 2 else ''})
        def list_waypoints(self):
            return ExecutionResult(True, 'fixture', {'stdout': 'names=[]', 'names': []})
    brain.read_backend = Backend()
    result = brain.run_text_task('机器人状态，然后查询点位', read_only_live=True)
    assert result['status'] == 'completed'
    assert [a['status'] for a in result['steps'][0]['attempts']] == ['failed', 'completed']
    events = brain.store.events(result['task_id'])
    assert [e['kind'] for e in events].count('recovery_reobserve') == 1
    assert all(s['effect_status'] == 'not_applicable' for s in result['steps'])
    assert brain.memory.get_stats()['memory']['total'] == 0


def test_parse_fallback_empty_list_is_not_valid_readback(brain):
    brain.read_backend = SimpleNamespace(list_waypoints=lambda: ExecutionResult(True, 'accepted', {'stdout': 'unparseable', 'names': []}))
    result = brain.run_steps([{'intent': 'list_waypoints'}, {'intent': 'robot_status'}], read_only_live=True)
    assert result['status'] == 'unverified'
    assert result['steps'][0]['failure_code'] == 'READ_RESPONSE_INVALID'
    assert result['steps'][1]['status'] == 'pending'


def test_malformed_backend_evidence_cannot_leave_orphan_running_task(brain):
    brain.read_backend = SimpleNamespace(robot_status=lambda:
        ExecutionResult(True, 'fixture', {'stdout': '/node', 'bad_number': float('nan')}))
    result = brain.run_text_task('机器人状态', read_only_live=True)
    assert result['status'] == 'failed'
    assert brain.get_task(result['task_id'])['status'] == 'failed'
    assert not brain.store.running()


def test_runtime_limp_blocks_wrong_order_before_next_read(brain):
    brain.read_backend = SimpleNamespace(robot_status=lambda: ExecutionResult(True, 'fixture', {'stdout': '/node'}),
        list_waypoints=lambda: ExecutionResult(True, 'fixture', {'stdout': 'names=[]', 'names': []}))
    constraint = {'formula': ['and', ['until', ['not', 'b'], 'a'], ['until', 'True', 'b']],
                  'propositions': {'robot_status': 'a', 'list_waypoints': 'b'}}
    wrong = brain.run_steps([{'intent': 'list_waypoints'}, {'intent': 'robot_status'}], read_only_live=True, constraint=constraint)
    assert wrong['steps'][0]['failure_code'] == 'TEMPORAL_CONSTRAINT_VIOLATED'
    assert wrong['steps'][0]['controller_status'] == 'not_sent'
    assert wrong['constraints']['trace'] == []
    right = brain.run_steps([{'intent': 'robot_status'}, {'intent': 'list_waypoints'}], read_only_live=True, constraint=constraint)
    assert right['status'] == 'completed'
    assert right['constraints']['accepted'] and right['constraints']['trace'] == ['a', 'b']
    unfinished = brain.run_steps([{'intent': 'robot_status'}], read_only_live=True, constraint=constraint)
    assert unfinished['status'] == 'unverified'
    assert unfinished['failure_code'] == 'TEMPORAL_CONSTRAINT_UNSATISFIED'


def test_simulation_and_motion_cannot_prove_runtime_propositions(brain):
    result = brain.run_steps([{'intent': 'robot_status'}], constraint={'formula': 'a', 'propositions': {'robot_status': 'a'}})
    assert result['failure_code'] == 'INVALID_CONSTRAINT'


def test_cancel_and_read_events_are_available_while_task_is_running(brain):
    entered, release = threading.Event(), threading.Event()
    def blocking_read():
        entered.set()
        assert release.wait(5)
        return ExecutionResult(True, 'fixture', {'stdout': '/test_node'})
    brain.read_backend = SimpleNamespace(robot_status=blocking_read)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(brain.run_text_task, '机器人状态，然后查询点位', False, True)
        try:
            assert entered.wait(3)
            running = brain.store.running()
            assert len(running) == 1
            task_id = running[0]['task_id']
            assert brain.get_task(task_id)['steps'][0]['status'] == 'running'
            assert brain.store.events(task_id)[-1]['kind'] == 'execution_started'
            assert brain.run_text_task('机器人状态')['failure_code'] == 'CONTROL_BUSY'
            reopened = UpstreamBrain(brain.original, runtime=brain.runtime, auto_summary=False)
            assert reopened.get_task(task_id)['status'] == 'running'
            cancelled = brain.cancel_task(task_id)
            assert cancelled['status'] == 'cancel_requested'
            assert cancelled['physical_stop_verified'] is False
        finally:
            release.set()
        result = future.result(timeout=3)
    assert result['status'] == 'cancelled'
    assert result['steps'][1]['status'] == 'pending'


def test_restart_preserves_unknown_effect_without_replay(brain):
    brain._save({'task_id': 'interrupted-test', 'status': 'running', 'text': 'fixture',
                 'steps': [{'intent': 'arm_preset', 'status': 'running'}]})
    reopened = UpstreamBrain(brain.original, runtime=brain.runtime, auto_summary=False)
    result = reopened.get_task('interrupted-test')
    assert result['status'] == 'interrupted'
    assert result['steps'][0]['effect_status'] == 'unknown'
    assert result['steps'][0]['failure_code'] == 'PROCESS_INTERRUPTED'


def test_legacy_text_asr_brain_and_openclaw_cannot_bypass_gateway(brain, monkeypatch):
    brain.config.openclaw_robot.enabled = True
    def forbidden(*args, **kwargs):
        pytest.fail('Independent OpenClaw controller constructor must not run')
    monkeypatch.setattr('zeng_agent.bootstrap.OpenClawRobotController', forbidden)
    app = create_app(brain.config, runtime=brain.runtime / 'http')
    app.state.upstream_brain.auto_summary = False
    client = TestClient(app)
    text = client.post('/command/text', json={'text': '左臂向上移动一点'}).json()
    assert text['result']['success'] is False
    assert text['task_status'] == 'blocked' and text['task_id']
    app.state.unified_service.local_asr_provider = SimpleNamespace(capture_and_transcribe=lambda:
        AsrResult(True, 'test-only transcription', CommandText('左臂向上移动一点', 'fixture')))
    assert client.post('/command/asr/local').json()['task_status'] == 'blocked'
    assert client.post('/brain/task/text', json={'text': '左臂向上移动一点'}).json()['status'] == 'blocked'
    assert client.post('/openclaw/robot/turn', json={'session_id': 'fixture', 'angle_deg': 30}).json()['success'] is False
    stop = client.post('/command/text', json={'text': '停止导航'}).json()
    assert stop['task']['physical_stop_verified'] is False
    assert stop['result']['success'] is False
    response = client.post('/upstream/task/steps', json={'steps': [{'intent': 'robot_status', 'slots': {'bad': 1}}]})
    assert response.json()['failure_code'] == 'INVALID_PARAMETERS'


def test_planner_only_receives_current_feasible_skills(brain, monkeypatch):
    from zeng_agent.llm_planner import LLMTaskPlanner
    captured = {}
    def post(url, **kwargs):
        captured.update(kwargs['json'])
        return SimpleNamespace(raise_for_status=lambda: None,
            json=lambda: {'choices': [{'message': {'content': '{"steps":[{"intent":"robot_status","slots":{}}]}'}}]})
    monkeypatch.setattr('zeng_agent.llm_planner.requests.post', post)
    brain.config.llm.enabled = True
    brain.original.llm_planner = LLMTaskPlanner(brain.config.llm)
    result = brain.run_text_task('检查一下当前状态')
    assert result['status'] == 'completed'
    prompt = captured['messages'][0]['content']
    assert 'robot_status' in prompt and 'observe_rgbd' in prompt
    assert 'arm_continuous_move' not in prompt and 'arm_stop' not in prompt


@pytest.mark.parametrize('steps', [[None], [{'intent': float('nan')}], None, 'invalid'])
def test_invalid_model_plan_is_durably_rejected(brain, steps):
    brain.config.llm.enabled = True
    brain.original.llm_planner = SimpleNamespace(plan=lambda text: {'steps': steps})
    result = brain.run_text_task('检查当前情况')
    assert result['status'] == 'blocked'
    assert result['failure_code'] == 'INVALID_PARAMETERS'
    assert brain.get_task(result['task_id']) == result


def test_reflect_worker_preserves_parent_cwd_and_refreshes_changed_evidence(brain):
    task = brain.run_text_task('左臂向上移动一点', confirmed=True)
    before = os.getcwd()
    first = summarize(task, brain.runtime)
    assert os.getcwd() == before
    changed = copy.deepcopy(task)
    changed['steps'][0]['message'] = 'new independent observation marker'
    second = summarize(changed, brain.runtime)
    assert first['directory'] != second['directory']
    assert 'new independent observation marker' in second['L1']
    assert os.getcwd() == before


@pytest.mark.parametrize('change', ['empty_velocity', 'nan_velocity', 'missing_trace', 'stale_recheck'])
def test_bad_physical_evidence_never_enters_memory(brain, change):
    if not os.environ.get('SEVA_ARCHIVED_CHASSIS_LOG'):
        pytest.skip('Optional archived-hardware test: recordings are not distributed')
    data = json.loads(Path(os.environ['SEVA_ARCHIVED_CHASSIS_LOG']).read_text(encoding='utf-8'))
    if change == 'empty_velocity':
        data['final_velocity'] = {}
    elif change == 'nan_velocity':
        data['final_velocity']['x'] = float('nan')
    elif change == 'missing_trace':
        data['trace'] = []
    else:
        data['independent_stationary_recheck']['odom_age_seconds'] = 10
    log = brain.runtime / 'injected.json'
    log.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        brain.import_verified_chassis_log(log)
    assert brain.memory.get_stats()['memory']['total'] == 0


def camera_fixture(advancing=True, age=0, mismatch=False):
    config = SimpleNamespace(CAMERA_SPECS={'rgbd_head_color': 'color', 'rgbd_head_depth': 'depth'})
    class Reader:
        EPOCH_NS_MIN, EPOCH_NS_MAX = 946684800 * 10**9, 4102444800 * 10**9
        count = 0
        def read_shm_metadata(self, spec):
            self.count += 1
            frame_id = (self.count + 1) // 2 if advancing else 1
            return (int((time.time()-age)*1e9), 2, 2, 3, 1, 12, 0, frame_id * 2, frame_id)
        def read_shm_frame(self, spec, meta):
            return SimpleNamespace(timestamp_ns=meta[0], spec=spec)
        def frame_to_hwc(self, frame, is_depth, rgb=False):
            return np.zeros((3 if mismatch and is_depth else 2, 2, 1 if is_depth else 3), dtype=np.uint16 if is_depth else np.uint8)
    return config, Reader()


@pytest.mark.parametrize('advancing,age', [(False, 0), (True, 30)])
def test_frozen_or_old_camera_cannot_create_fresh_observation(tmp_path, advancing, age):
    result = capture_rgbd(tmp_path, timeout=.04, reader_pair=camera_fixture(advancing, age))
    assert result.success is False and result.data['failure_code'] == 'OBSERVATION_STALE'
    assert not list(tmp_path.glob('*.npz'))


def test_fresh_rgbd_has_artifact_but_no_assumed_metric_target(tmp_path):
    result = capture_rgbd(tmp_path, timeout=.1, reader_pair=camera_fixture())
    assert result.success
    assert Path(result.data['artifact']).exists()
    assert result.data['world_pose_available'] is False
    assert result.data['metric_target_positions'] is None


def test_mismatched_rgbd_is_invalid(tmp_path):
    result = capture_rgbd(tmp_path, timeout=.1, reader_pair=camera_fixture(mismatch=True))
    assert result.data['failure_code'] == 'OBSERVATION_INVALID'
