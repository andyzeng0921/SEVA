"""Route existing HTTP, ASR and relay service methods through one task owner."""
from dataclasses import asdict
from zeng_agent.models import CommandText
from zeng_agent.service import RobotAgentService


class UnifiedService(RobotAgentService):
    def __init__(self, original, brain):
        super().__init__(config=original.config, waypoint_names=original._waypoint_names,
            executor=original.executor, parser=original.parser,
            local_asr_provider=original.local_asr_provider, brain_orchestrator=brain,
            openclaw_controller=None)

    def handle_text_command(self, text, source='text', confidence=1.0):
        command = CommandText(text=text, source=source, confidence=confidence)
        task = self.brain_orchestrator.run_text_task(text, confirmed=False)
        return {'command': asdict(command), 'parsed': asdict(self.parser.parse(text)),
            'task_id': task.get('task_id'), 'task_status': task['status'], 'task': task,
            'result': {'success': task['status'] == 'completed',
                'message': task.get('message', task['status']), 'latency_ms': 0.0,
                'data': {'mode': 'unified_task_gateway', 'physical_stop_verified': False}}}
