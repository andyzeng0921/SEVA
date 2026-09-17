import argparse
import json
import os
from zeng_agent.bootstrap import build_service
from zeng_agent.config import AgentConfig
from .source import PROJECT
from .bridge import UpstreamBrain
from .reflect_adapter import summarize

parser = argparse.ArgumentParser(description='Pinned upstream algorithms, 306 adapters')
parser.add_argument('command', choices=['status', 'task', 'summary', 'reflect', 'import-evidence'])
parser.add_argument('--text')
parser.add_argument('--task-id')
parser.add_argument('--evidence')
parser.add_argument('--read-only-live', action='store_true')
args = parser.parse_args()
config = AgentConfig.from_yaml(os.environ.get('ZENG_CONFIG', str(PROJECT / 'config.yaml')))
config.openclaw_robot.enabled = False
brain = UpstreamBrain(build_service(config).brain_orchestrator, runtime=os.environ.get('ZENG_RUNTIME'))
if args.command == 'status':
    result = brain.status()
elif args.command == 'task':
    result = brain.run_text_task(args.text, read_only_live=args.read_only_live)
elif args.command == 'import-evidence':
    result = brain.import_verified_chassis_log(args.evidence)
else:
    task = brain.get_task(args.task_id)
    if task is None:
        raise SystemExit('Unknown task id')
    result = summarize(task, brain.runtime, config.llm if args.command == 'reflect' else None)
print(json.dumps(result, ensure_ascii=False, indent=2))
