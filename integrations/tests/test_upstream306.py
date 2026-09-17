import inspect
import os
from pathlib import Path
import pytest
from zeng_agent.bootstrap import build_service
from zeng_agent.config import AgentConfig
from zeng_agent.models import ExecutionResult
from upstream306.bridge import UpstreamBrain, check_trace, SkillOperation
from upstream306.reflect_adapter import summarize
from upstream306.source import PROJECT


@pytest.fixture
def brain(tmp_path):
    cfg = AgentConfig.from_yaml(PROJECT / 'config.yaml')
    cfg.dry_run = True
    cfg.llm.enabled = False
    cfg.openclaw_robot.enabled = False
    return UpstreamBrain(build_service(cfg).brain_orchestrator, runtime=tmp_path, auto_summary=False)


def test_uses_upstream_algorithm(brain):
    assert 'upstream/stretch_ai/src/stretch/core/task.py' in Path(inspect.getfile(SkillOperation.__bases__[0])).as_posix()
    assert 'physmem/learning/loop.py' in Path(inspect.getfile(type(brain.memory))).as_posix()


def test_status_chain_and_durable_task(brain):
    result = brain.run_text_task('机器人状态，然后查询点位')
    assert result['status'] == 'completed'
    assert len(result['steps']) == 2
    reopened = UpstreamBrain(brain.original, runtime=brain.runtime)
    assert reopened.get_task(result['task_id']) == result
    assert reopened.memory.get_stats()['memory']['total'] == 0


def test_motion_requires_confirmation(brain):
    result = brain.run_text_task('左臂向上移动一点')
    assert result['status'] == 'blocked'
    assert result['steps'][0]['controller_success'] is None


def test_accepted_dry_motion_never_claims_effect(brain):
    result = brain.run_text_task('左臂向上移动一点，然后机器人状态', confirmed=True)
    assert result['status'] == 'unverified'
    assert result['steps'][0]['controller_success'] is True
    assert result['steps'][0]['effect_status'] == 'unknown'
    assert result['steps'][1]['status'] == 'pending'
    assert brain.memory.get_stats()['memory']['total'] == 0


def test_read_only_live_cannot_fall_through(brain):
    result = brain.run_text_task('机器人状态，然后左臂向上移动一点', confirmed=True, read_only_live=True)
    assert result['status'] == 'blocked'
    assert result['steps'] == []


def test_failed_read_has_one_bounded_recovery_then_stops(brain):
    class Failure:
        count = 0
        def robot_status(self):
            self.count += 1
            return ExecutionResult(False, 'Injected test read failure')
    backend = Failure()
    brain.read_backend = backend
    result = brain.run_text_task('机器人状态，然后查询点位', read_only_live=True)
    assert result['status'] == 'failed'
    assert result['steps'][1]['status'] == 'pending'
    assert backend.count == 2
    assert len(result['steps'][0]['attempts']) == 2


def test_limp_sequence_and_unsafe_precedence():
    # a must remain absent until b; b then permits a on a later step.
    formula = ['and', ['until', ['not', 'a'], 'b'],
               ['until', 'True', ['and', 'b', ['next', ['until', 'True', 'a']]]]]
    assert check_trace(formula, ['', 'b', 'a'])['accepted']
    assert check_trace(formula, ['a', 'b'])['violated']
    with pytest.raises(ValueError):
        check_trace(['always', 'a'], ['a'])


def test_reflect_original_summary_and_failure_branch(brain):
    task = brain.run_text_task('左臂向上移动一点', confirmed=True)
    class ReplayPrompter:
        # Deterministic transport fixture; these are NOT live model outputs.
        answers = iter(['No, the physical effect is unverified',
                        'The controller accepted a simulated command, with no physical observation.',
                        '00:01'])
        def query(self, **kwargs):
            return next(self.answers), None
    output = summarize(task, brain.runtime, prompter=ReplayPrompter())
    assert 'Visual observation:' in output['L1']
    assert 'unknown' in output['L1']
    assert 'Goal:' in output['L2']
    assert output['reasoning']['pred_failure_step'] == ['00:01']


def test_prior_real_evidence_persistence_and_dedup(brain):
    if not os.environ.get('SEVA_ARCHIVED_CHASSIS_LOG'):
        pytest.skip('Optional archived-hardware test: recordings are not distributed')
    evidence = Path(os.environ['SEVA_ARCHIVED_CHASSIS_LOG'])
    result = brain.import_verified_chassis_log(evidence)
    assert brain.import_verified_chassis_log(evidence)['duplicate']
    reopened = UpstreamBrain(brain.original, runtime=brain.runtime)
    assert reopened.memory.get_stats()['memory']['total'] == 1
    assert reopened.memory.get_stats()['principles']['total'] == 0


def test_bumble_original_prompt_is_context_only(brain):
    result = brain.bumble_context('检查机器人状态')
    assert 'TASK DESCRIPTION' in result['task_prompt']
    assert result['execution_enabled'] is False
