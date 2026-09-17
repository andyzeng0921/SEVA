"""Adapt zeng-agent skills to the upstream Stretch AI Task/Operation protocol."""
from dataclasses import asdict
import copy
import json
import math
from pathlib import Path
import threading
import uuid
from filelock import FileLock, Timeout

from zeng_agent.models import ExecutionRequest, ExecutionResult, utc_now_iso
from zeng_agent.backends import Ros2ShellBackend
from zeng_agent.brain import BrainStep
from zeng_agent.skill_packages.catalog import SkillPackage
from .source import PROJECT, stretch_task, physmem, limp, bumble_prompts
from .contracts import READ_ONLY, normalize, schema_for
from .events import EventStore
from .observations import capture_rgbd

_task_module = stretch_task()
RETRYABLE_READ_FAILURES = {'READ_FAILED', 'READ_RESPONSE_INVALID', 'OBSERVATION_UNAVAILABLE', 'OBSERVATION_STALE'}


class SkillOperation(_task_module.Operation):
    def __init__(self, step, bridge, task_state, confirmed, read_only_live, initial_error=None):
        super().__init__(f"step_{step.index}_{step.intent}", max_failures=1)
        self.step, self.bridge = step, bridge
        self.confirmed, self.read_only_live = confirmed, read_only_live
        self.task_state, self.initial_error = task_state, initial_error
        self.record = {'index': step.index, 'intent': step.intent, 'slots': step.slots,
                       'controller_success': None, 'effect_status': 'unknown',
                       'controller_status': 'not_sent', 'failure_code': None,
                       'status': 'pending', 'evidence_ids': [], 'attempts': []}

    def event(self, kind, payload=None):
        evidence_id = self.bridge.store.append(self.task_state['task_id'], kind,
            payload or {k: v for k, v in self.record.items() if k not in {'attempts', 'evidence_ids'}})
        self.record['evidence_ids'].append(evidence_id)
        self.record['evidence_id'] = evidence_id
        self.bridge._save(self.task_state)

    def cancelled(self):
        return self.bridge.store.cancelled(self.task_state['task_id'])

    def can_start(self):
        reason, code = self.bridge.guard(self.step.intent, self.confirmed, self.read_only_live)
        if self.initial_error:
            reason, code = self.initial_error, 'INVALID_PARAMETERS'
        if self.cancelled():
            reason, code = 'Task cancellation requested', 'CANCEL_REQUESTED'
        constraint = self.task_state.get('constraints')
        if reason is None and constraint:
            valuation = constraint['propositions'].get(self.step.intent, '')
            projected = check_trace(constraint['formula'], constraint['trace'] + [valuation])
            if projected['violated']:
                reason, code = 'Verified-step ordering would violate the requested constraint', 'TEMPORAL_CONSTRAINT_VIOLATED'
        if reason:
            self.record.update(status='cancelled' if code == 'CANCEL_REQUESTED' else 'blocked',
                               message=reason, failure_stage='precondition', failure_code=code)
            self.event('precondition_rejected')
        else:
            self.event('precondition_passed')
        return reason is None

    def run(self):
        self._started = True
        self.record.update(status='running', controller_status='executing', failure_code=None)
        self.record.pop('failure_stage', None)
        self.event('execution_started')
        try:
            if self.step.intent == 'observe_rgbd':
                result = (self.bridge.observer(self.bridge.runtime / 'observations') if self.read_only_live else
                          ExecutionResult(True, 'Observation simulation; no camera evidence', {'mode': 'dry_run'}))
                mode = 'read_only_live' if self.read_only_live else 'dry_run'
            elif self.read_only_live and self.step.intent in READ_ONLY:
                result = getattr(self.bridge.read_backend, self.step.intent)()
                mode = 'read_only_live'
            else:
                mode = 'dry_run' if self.bridge.config.dry_run else 'live'
                result = self.bridge.original.executor.execute(ExecutionRequest(
                    self.step.intent, self.step.slots, mode))
            # Malformed backend output cannot corrupt the durable task record.
            if not isinstance(result.success, bool) or not isinstance(result.data, dict):
                raise ValueError('Invalid controller response shape')
            json.dumps(asdict(result), allow_nan=False)
            self.record.update(result=asdict(result), controller_success=result.success,
                               mode=mode, message=result.message,
                               controller_status='completed' if self.step.intent in READ_ONLY and result.success else
                                                 'accepted' if result.success else 'failed')
            # The old backend reports acceptance, not observed effects. Never turn
            # an accepted command or dry run into a verified physical success.
            verified = (result.success and self.step.intent in READ_ONLY and
                        (mode != 'read_only_live' or self._valid_readback(result)))
            self.record['effect_status'] = ('not_applicable' if verified else
                'failed' if self.step.intent in READ_ONLY and not result.success else 'unknown')
            self.record['verification'] = ('read_response_valid' if mode == 'read_only_live' else 'dry_run_only') if verified else 'not_verified'
            self.record['status'] = 'completed' if verified else ('failed' if not result.success else 'unverified')
            if not verified:
                self.record['failure_stage'] = 'execution' if not result.success else 'postcondition'
                self.record['failure_code'] = (result.data.get('failure_code', 'READ_FAILED' if self.step.intent in READ_ONLY else 'CONTROLLER_FAILED')
                                              if not result.success else 'READ_RESPONSE_INVALID' if self.step.intent in READ_ONLY else 'EFFECT_UNVERIFIED')
        except Exception as exc:
            self.record.update(status='failed', controller_success=False, effect_status='unknown',
                               controller_status='failed', failure_code='READ_FAILED' if self.step.intent in READ_ONLY else 'CONTROLLER_FAILED',
                               failure_stage='execution', message=type(exc).__name__ + ': ' + str(exc))
        if self.cancelled():
            self.record.update(status='cancelled', failure_code='CANCEL_REQUESTED', failure_stage='execution',
                               effect_status='unknown', message='Cancellation observed; subsequent skills will not execute')
        constraint = self.task_state.get('constraints')
        if self.record['status'] == 'completed' and constraint:
            constraint['trace'].append(constraint['propositions'].get(self.step.intent, ''))
            checked = check_trace(constraint['formula'], constraint['trace'])
            constraint.update(accepted=checked['accepted'], violated=checked['violated'])
            self.event('temporal_constraint_checked', {'trace': constraint['trace'],
                'accepted': checked['accepted'], 'violated': checked['violated']})
        self.record['attempts'].append(copy.deepcopy({k: v for k, v in self.record.items() if k not in {'attempts', 'evidence_ids'}}))
        self.event('effect_checked')

    def _valid_readback(self, result):
        if self.step.intent == 'list_waypoints':
            names = result.data.get('names')
            # The shell backend otherwise turns a failed parse into an empty list.
            stdout = result.data.get('stdout', '')
            return isinstance(names, list) and all(isinstance(n, str) for n in names) and 'names=' in stdout
        if self.step.intent == 'observe_rgbd':
            return bool(result.data.get('observation_id') and result.data.get('sha256'))
        # robot_status is a ROS graph read, not a robot-health measurement.
        return bool(result.data.get('stdout', '').strip())

    def was_successful(self):
        return self.record['status'] == 'completed'


class FailureRecordOperation(_task_module.Operation):
    """Terminal adapter: preserve failed/blocked outcome after upstream routing."""
    def can_start(self):
        return True

    def run(self):
        self._started = True

    def was_successful(self):
        return False


class RefreshReadOperation(_task_module.Operation):
    """One explicit recovery edge in the upstream graph; never retry motion."""
    def __init__(self, original):
        super().__init__(original.name + '_refresh_once', max_failures=2)
        self.original = original

    def can_start(self):
        op = self.original
        return (op.step.intent in READ_ONLY and op.read_only_live and not op.cancelled()
                and op.record.get('failure_code') in RETRYABLE_READ_FAILURES)

    def run(self):
        self._started = True
        self.original.event('recovery_reobserve', {'intent': self.original.step.intent, 'retry_budget': 1})
        self.original.run()

    def was_successful(self):
        return self.original.was_successful()


class UpstreamBrain:
    def __init__(self, original, runtime=None, read_backend=None, observer=None, auto_summary=True):
        self.original, self.config = original, original.config
        self.runtime = Path(runtime or PROJECT / 'runtime/upstream306')
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.db = self.runtime / 'tasks.sqlite3'
        self.lock = threading.RLock()
        self.lease = FileLock(str(self.runtime / 'execution.lock'))
        self.observer = observer or capture_rgbd
        self.auto_summary = auto_summary
        self.read_backend = read_backend or Ros2ShellBackend(config=self.config)
        self.store = EventStore(self.db)
        self.catalog = original.skill_catalog
        if self.catalog.by_intent('observe_rgbd') is None:
            self.catalog.register(SkillPackage('observe_rgbd', 'perception', 'Capture a fresh synchronized RGB-D observation without motion.', 'observe_rgbd'))
        try:
            with self.lease.acquire(timeout=0):
                for interrupted in self.store.running():
                    interrupted.update(status='interrupted', message='Execution owner exited; no automatic motion replay')
                    for step in interrupted.get('steps', []):
                        if step.get('status') == 'running':
                            step.update(status='unverified', effect_status='unknown', failure_code='PROCESS_INTERRUPTED')
                    self.store.append(interrupted['task_id'], 'process_interrupted', {'effect_status': 'unknown'})
                    self._save(interrupted)
        except Timeout:
            pass  # An existing execution owner still holds the process-wide lease.
        p = physmem()
        memory_dir = self.runtime / 'physical_memory'
        cfg = p.ScientificLearningConfig(memory_name='robot306_verified_physical',
                save_path=str(memory_dir), run_consolidation_async=False)
        self.memory = (p.PhysMem.load_state(str(memory_dir), config=cfg)
                       if (memory_dir / 'memory.json').exists() else p.PhysMem(config=cfg))

    def status(self):
        previous = {k: v for k, v in self.original.status().items()
                    if k not in {'skills', 'skill_packages', 'tasks_in_memory', 'require_confirmation_for_motion'}}
        contracts = self.capabilities()
        return {**previous, 'framework': 'stretch_ai_upstream_task',
                'robot_topic_node_id': self.config.topic_node_id,
                'require_confirmation_for_motion': True, 'skills': contracts, 'skill_packages': contracts,
                'control_scope': 'this gateway and its execution lease only; external ROS publishers are not fenced',
                'state_backend': 'sqlite', 'algorithm_file': _task_module.__file__,
                'upstream_motion_commissioned': False,
                'physical_memory': self.memory.get_stats(),
                'contracts': contracts, 'active_tasks': [t['task_id'] for t in self.store.running()],
                'runtime': str(self.runtime)}

    def _save(self, payload):
        return self.store.save(payload)

    def get_task(self, task_id):
        return self.store.get(task_id)

    def guard(self, intent, confirmed=False, read_only_live=False):
        package = self.catalog.by_intent(intent)
        if package is None:
            return 'Unsupported skill', 'UNSUPPORTED_SKILL'
        if intent == 'arm_stop':
            return 'Existing arm stop is mapped to reset; true stop semantics are not commissioned', 'ARM_STOP_UNVERIFIED'
        if read_only_live and intent not in READ_ONLY:
            return 'read_only_live accepts observation/status/list only', 'READ_ONLY_VIOLATION'
        if not self.config.dry_run and intent not in READ_ONLY:
            return '306 motion adapter has not passed task-level commissioning', 'HARDWARE_UNCOMMISSIONED'
        if intent not in READ_ONLY and not confirmed and not package.safety_override:
            return 'Action requires confirmation', 'CONFIRMATION_REQUIRED'
        return None, None

    def capabilities(self, confirmed=False, read_only_live=False):
        output = []
        busy = bool(self.store.running())
        for package in self.catalog.list():
            reason, code = self.guard(package.intent, confirmed, read_only_live)
            description = ('Cancel this gateway task; physical navigation stop remains unverified' if package.intent == 'stop_navigation'
                           else 'Unavailable: existing arm reset is not a verified stop' if package.intent == 'arm_stop' else package.description)
            output.append({'intent': package.intent, 'description': description, 'parameters': schema_for(package),
                'available_in_mode': reason is None and not busy, 'live_adapter_bound': package.intent in READ_ONLY,
                'mode': 'read_only_live' if read_only_live else 'dry_run' if self.config.dry_run else 'live',
                'reason': 'Another task holds the execution lease' if busy else reason,
                'failure_code': 'CONTROL_BUSY' if busy else code,
                'preconditions': ['validated_parameters', 'execution_lease', 'authorization'],
                'postconditions': ['valid_read_response' if package.intent in READ_ONLY else 'physical_effect_evidence_required']})
        return output

    def cancel_task(self, task_id):
        task = self.get_task(task_id)
        if task is None:
            return {'status': 'not_found', 'task_id': task_id}
        if task['status'] != 'running':
            return {'status': 'already_terminal', 'task_id': task_id, 'task_status': task['status'], 'physical_stop_verified': False}
        self.store.cancel(task_id)
        evidence_id = self.store.append(task_id, 'cancel_requested', {'physical_stop_verified': False})
        return {'status': 'cancel_requested', 'task_id': task_id, 'evidence_ids': [evidence_id],
                'physical_stop_verified': False, 'message': 'Cancellation queued; no claim about external ROS controllers'}

    def cancel_active(self):
        active = self.store.running()
        return {'status': 'cancel_requested' if active else 'no_owned_task',
                'tasks': [self.cancel_task(t['task_id']) for t in active], 'physical_stop_verified': False}

    def run_text_task(self, text, confirmed=False, read_only_live=False):
        try:
            if self.original.parser.parse(text).intent == 'stop_navigation':
                return self.cancel_active()
            if text.strip() in {'观察环境', '查看相机', '观察RGBD'}:
                steps = [{'intent': 'observe_rgbd', 'slots': {}}]
            elif self.original.llm_planner is not None and self.config.llm.enabled:
                # Reuse the existing planner transport/parser with a per-request
                # tool list, rather than mutating its shared static prompt.
                planner = copy.copy(self.original.llm_planner)
                allowed = [c for c in self.capabilities(confirmed, read_only_live)
                           if c['available_in_mode'] and c['intent'] != 'stop_navigation']
                planner._build_prompt = lambda: (
                    '将任务转换为 JSON {"steps":[{"intent":"技能名称","slots":{}}]}。'
                    '只能选择以下当前允许的技能和参数；不可执行时返回空 steps。'
                    '模拟模式不代表已在实机完成。技能：' + json.dumps(allowed, ensure_ascii=False) +
                    '允许预设：' + json.dumps(self.config.allowed_arm_presets, ensure_ascii=False) +
                    '已知航点：' + json.dumps(self.config.waypoint_names, ensure_ascii=False))
                steps = planner.plan(text).get('steps', [])
            else:
                steps = [asdict(step) for step in self.original._plan(self.original._split_text(text))]
        except Exception as exc:
            return self._save({'task_id': str(uuid.uuid4()), 'text': text, 'created_at': utc_now_iso(),
                'status': 'failed', 'steps': [], 'failure_code': 'PLANNING_FAILED', 'message': type(exc).__name__})
        if isinstance(steps, list) and len(steps) == 1 and isinstance(steps[0], dict) and steps[0].get('intent') == 'stop_navigation':
            return self.cancel_active()
        return self.run_steps(steps, text, confirmed, read_only_live)

    def run_steps(self, step_specs, text='Structured skill request', confirmed=False, read_only_live=False, constraint=None):
        result = {'task_id': str(uuid.uuid4()), 'text': text, 'created_at': utc_now_iso(),
                  'status': 'blocked', 'steps': [], 'framework': 'stretch_ai_upstream_task'}
        if not isinstance(step_specs, list) or any(not isinstance(s, dict) or not isinstance(s.get('intent'), str) for s in step_specs):
            return self._save({**result, 'message': 'steps must be an array of objects', 'failure_code': 'INVALID_PARAMETERS'})
        if constraint is not None:
            try:
                if not read_only_live or not isinstance(constraint, dict):
                    raise ValueError('Runtime constraints currently require read_only_live evidence')
                formula, propositions = constraint['formula'], constraint['propositions']
                if (not isinstance(propositions, dict) or not propositions or
                    any(k not in READ_ONLY or not isinstance(v, str) or len(v) != 1 or v not in 'abcdef'
                        for k, v in propositions.items()) or len(set(propositions.values())) != len(propositions)):
                    raise ValueError('Propositions must uniquely map read-only skills to atoms a-f')
                initial = check_trace(formula, [])
                if initial['violated']:
                    raise ValueError('Constraint is already violated')
                result['constraints'] = {'formula': formula, 'propositions': propositions, 'trace': [],
                    'accepted': initial['accepted'], 'violated': False, 'source': initial['source'],
                    'evidence_semantics': 'validated live read responses only; no inferred physical propositions'}
            except (ValueError, KeyError, TypeError) as exc:
                return self._save({**result, 'message': str(exc), 'failure_code': 'INVALID_CONSTRAINT'})
        if len(step_specs) == 1 and step_specs[0].get('intent') == 'stop_navigation' and not step_specs[0].get('slots'):
            return self.cancel_active()
        try:
            lease = self.lease.acquire(timeout=0)
        except Timeout:
            return self._save({**result, 'message': 'Another task holds the execution lease', 'failure_code': 'CONTROL_BUSY'})
        with lease:
            if not self.config.brain.enabled:
                return self._save({**result, 'message': 'Brain disabled'})
            if not step_specs or len(step_specs) > self.config.brain.max_steps:
                return self._save({**result, 'message': 'Invalid plan length'})
            # Read-only live mode is an allowlist for the WHOLE request; it must
            # never silently fall through to a simulated or live motion branch.
            if read_only_live and any(s.get('intent') not in READ_ONLY for s in step_specs):
                return self._save({**result, 'message': 'read_only_live accepts observation/status/list only', 'failure_code': 'READ_ONLY_VIOLATION'})
            steps, errors = [], []
            for index, item in enumerate(step_specs, 1):
                intent = item.get('intent', 'unsupported')
                package = self.catalog.by_intent(intent)
                try:
                    if package is None:
                        raise ValueError('Unsupported skill')
                    slots = normalize(package, item.get('slots', {}))
                    if intent == 'arm_preset' and slots['action_name'] not in self.config.allowed_arm_presets:
                        raise ValueError('Arm preset not allowed')
                    if intent == 'navigate_to_waypoint' and self.config.waypoint_names and slots['name'] not in self.config.waypoint_names:
                        raise ValueError('Unknown waypoint')
                    error = None
                except ValueError as exc:
                    slots, error = {}, str(exc)
                steps.append(BrainStep(index, item.get('text', ''), intent, slots, package.permission if package else 'safe_action'))
                errors.append(error)
            # Parameter errors in a later step must not allow earlier side effects.
            if any(errors):
                result.update(failure_code='INVALID_PARAMETERS', message=next(e for e in errors if e),
                    steps=[{'index': s.index, 'intent': s.intent, 'slots': s.slots, 'status': 'blocked' if e else 'pending',
                            'controller_success': None, 'controller_status': 'not_sent', 'effect_status': 'unknown',
                            'failure_code': 'INVALID_PARAMETERS' if e else None, 'message': e} for s, e in zip(steps, errors)])
                self.store.append(result['task_id'], 'plan_rejected', {'errors': errors})
                return self._save(result)
            task = _task_module.Task(max_failures=1)
            operations = [SkillOperation(s, self, result, confirmed, read_only_live) for s in steps]
            for op in operations:
                task.add_operation(op)
            failure = FailureRecordOperation('record_failure', max_failures=2)
            task.add_operation(failure, terminal=True)
            for op in operations:
                task.connect_on_cannot_start(op.name, failure.name)
                retry = RefreshReadOperation(op)
                next_operation = op.on_success
                task.add_operation(retry, terminal=True)
                retry.on_success, retry.on_failure, retry.on_cannot_start = next_operation, failure, failure
                task.connect_on_failure(op.name, retry.name)
            # Save a pending record before entering any upstream blocking call.
            result.update(status='running', steps=[o.record for o in operations])
            self._save(result)
            self.store.append(result['task_id'], 'task_started', {'mode': 'read_only_live' if read_only_live else 'dry_run'})
            try:
                ok = task.run()
            except Exception as exc:
                ok = False
                self.store.append(result['task_id'], 'kernel_exception', {'exception_type': type(exc).__name__})
                for op in operations:
                    if op.record['status'] == 'running':
                        op.record.update(status='unverified', effect_status='unknown', failure_code='KERNEL_EXCEPTION')
            result['steps'] = [o.record for o in operations]
            if ok and result.get('constraints') and not result['constraints']['accepted']:
                ok = False
                result['failure_code'] = 'TEMPORAL_CONSTRAINT_UNSATISFIED'
            result['status'] = ('completed' if ok else
                                ('cancelled' if self.store.cancelled(result['task_id']) else
                                 'blocked' if any(o.record['status'] == 'blocked' for o in operations) else
                                 'unverified' if any(o.record['status'] == 'unverified' for o in operations) else 'failed'))
            if result.get('failure_code') == 'TEMPORAL_CONSTRAINT_UNSATISFIED':
                result['status'] = 'unverified'
            result['message'] = 'Read/simulation checks completed' if ok else 'Stopped at first unverified, blocked, or failed skill'
            result['updated_at'] = utc_now_iso()
            self.store.append(result['task_id'], 'task_finished', {'status': result['status']})
            self._save(result)
            if self.auto_summary and result['status'] in {'failed', 'unverified', 'cancelled'}:
                from .reflect_adapter import summarize
                try:
                    summary = summarize(result, self.runtime)
                    result['summary'] = {'directory': summary.get('directory'), 'source': summary.get('source')}
                except Exception as exc:
                    result['summary'] = {'status': 'unavailable', 'error_type': type(exc).__name__}
            return self._save(result)

    def bumble_context(self, text, history=None):
        upstream = bumble_prompts()
        allowed = [s['intent'] for s in self.capabilities(read_only_live=True) if s['available_in_mode']]
        skills = [f'{s.intent}: {s.description}' for s in self.catalog.list() if s.intent in allowed]
        instructions, prompt = upstream['make_prompt'](skills, text,
                info={'floor_num': 'unknown (306 is not the published TIAGo building)',
                      'add_obj_ind': False, 'step_idx': len(history or [])})
        return {'source': 'BUMBLE original make_prompt / make_history_prompt',
                'instructions': instructions, 'task_prompt': prompt,
                'history': upstream['make_history_prompt'](history or []),
                'allowed_skills': allowed,
                'physical_knowledge': {'established_principles': self.memory.get_principles_prompt(),
                    'model_hypotheses': self.memory.get_active_hypotheses_prompt(),
                    'scope': 'Historical evidence only; never relax permissions, limits or freshness checks'},
                'execution_enabled': False,
                'limitations': 'Original building/reach examples are TIAGo-specific; this endpoint only prepares context.'}

    def import_verified_chassis_log(self, log_path):
        """Map the already completed, bounded 306 experiment into PhysMem."""
        with self.lock, FileLock(str(self.runtime / 'memory.lock')):
            return self._import_verified_chassis_log(log_path)

    def _import_verified_chassis_log(self, log_path):
        import hashlib
        data_bytes = Path(log_path).read_bytes()
        data = json.loads(data_bytes)
        digest = hashlib.sha256(data_bytes).hexdigest()
        index_path = self.runtime / 'imported_evidence.json'
        seen = json.loads(index_path.read_text()) if index_path.exists() else {}
        if digest in seen:
            return {'duplicate': True, 'experience_id': seen[digest]}
        def finite(value):
            return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        velocity = data.get('final_velocity', {})
        recheck = data.get('independent_stationary_recheck', {})
        trace = data.get('trace', [])
        if not (data.get('robot_id') == 306 and data.get('execute') is True and
                data.get('outcome') == 'completed_and_stationary' and
                data.get('target_forward_m') == .1 and data.get('max_speed_mps') == .04 and
                finite(data.get('measured_forward_m')) and .08 <= data['measured_forward_m'] <= .12 and
                all(finite(velocity.get(k)) and abs(velocity[k]) <= .005 for k in ['x', 'y', 'angular_z']) and
                all(finite(recheck.get(k)) and abs(recheck[k]) <= .005
                    for k in ['displacement_during_2s_m', 'linear_x', 'linear_y', 'angular_z']) and
                finite(recheck.get('odom_age_seconds')) and 0 <= recheck['odom_age_seconds'] <= .3 and
                isinstance(trace, list) and len(trace) >= 3 and
                all(isinstance(t, dict) and all(finite(t.get(k)) for k in
                    ['t', 'forward_m', 'lateral_m', 'yaw_delta_rad', 'speed_command', 'odom_speed', 'clearance_m'])
                    and 0 <= t['speed_command'] <= .040001 and t['clearance_m'] >= .65
                    for t in trace)):
            raise ValueError('Evidence is not a successful real 306 experiment')
        # Reload under a process-wide memory lock to avoid overwriting another
        # CLI import with an older in-memory snapshot.
        memory_dir = self.runtime / 'physical_memory'
        if (memory_dir / 'memory.json').exists():
            self.memory = physmem().PhysMem.load_state(str(memory_dir), config=self.memory.config)
        eid, surprising = self.memory.record_experience(action='bounded_chassis_forward', success=True,
            symbolic_state={'robot_id': 306, 'evidence_scope': 'single_10cm_trial', 'source_sha256': digest,
                            'provenance': 'operator_imported_historical_log', 'generalization_allowed': False},
            extra_metrics={'source': str(log_path), 'evidence': data, 'effect_verified': True})
        self.memory.end_episode(success=True)
        self.memory.save_state()
        seen[digest] = eid
        index_path.write_text(json.dumps(seen, indent=2))
        return {'experience_id': eid, 'surprising': surprising, 'source_sha256': digest,
                'claim': 'One successful trial; no generalized physical rule established'}


def check_trace(formula, trace):
    """JSON-to-tuple protocol conversion around LIMP's original progression."""
    count = [0]
    def convert(node, depth=0):
        count[0] += 1
        if depth > 8 or count[0] > 48:
            raise ValueError('Formula exceeds adapter bound')
        if isinstance(node, str) and (node in {'True', 'False'} or len(node) == 1 and node in 'abcdef'):
            return node
        if isinstance(node, (tuple, list)) and node:
            arity = {'not': 2, 'next': 2, 'and': 3, 'or': 3, 'until': 3}.get(node[0])
            if arity == len(node):
                return tuple([node[0]] + [convert(x, depth + 1) for x in node[1:]])
        raise ValueError('Unsupported co-safe LTL formula; atoms a-f; always is not exposed')
    current = convert(formula)
    if len(trace) > 128 or any(not isinstance(t, str) or set(t) - set('abcdef') for t in trace):
        raise ValueError('Invalid bounded trace')
    upstream = limp()
    states = [current]
    for valuation in trace:
        current = upstream._progress(current, valuation)
        states.append(current)
    return {'source': upstream.__file__, 'states': states,
            'accepted': current == 'True', 'violated': current == 'False'}
