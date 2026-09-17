"""Mount upstream adapters into the existing zeng-agent HTTP application."""
import os
import copy
from fastapi import HTTPException
from pydantic import BaseModel, Field
from zeng_agent.api import create_app as original_app
from zeng_agent.bootstrap import build_service
from zeng_agent.config import AgentConfig
from .source import PROJECT, UPSTREAM
from .bridge import UpstreamBrain, check_trace
from .reflect_adapter import summarize
from .gateway import UnifiedService
import json


class TaskRequest(BaseModel):
    text: str = Field(min_length=1, max_length=6000)
    confirmed: bool = False
    read_only_live: bool = False


class TraceRequest(BaseModel):
    formula: list | str
    trace: list[str] = Field(max_length=128)


class StepRequest(BaseModel):
    intent: str = Field(min_length=1, max_length=64)
    slots: dict = Field(default_factory=dict)


class StructuredTaskRequest(BaseModel):
    steps: list[StepRequest] = Field(min_length=1, max_length=8)
    text: str = Field(default='Structured skill request', max_length=6000)
    confirmed: bool = False
    read_only_live: bool = False
    constraint: dict | None = None


def create_app(config=None, runtime=None, read_backend=None, observer=None):
    config = copy.deepcopy(config or AgentConfig.from_yaml(os.environ.get('ZENG_CONFIG', str(PROJECT / 'config.yaml'))))
    # Existing legacy endpoints have no unified live-motion gate; keep this
    # migration service in dry-run configuration until they are commissioned.
    if not config.dry_run:
        raise RuntimeError('Upstream306 HTTP service requires dry_run=true; live readback uses an explicit allowlist')
    # Disable the independent controller before its constructor can start any
    # hardware helpers. The legacy routes remain explicit disabled responses.
    config.openclaw_robot.enabled = False
    original = build_service(config)
    brain = UpstreamBrain(original.brain_orchestrator, runtime or os.environ.get('ZENG_RUNTIME'), read_backend, observer)
    service = UnifiedService(original, brain)
    app = original_app(service)
    app.state.upstream_brain = brain
    app.state.unified_service = service

    @app.get('/upstream/status')
    def status():
        return {**brain.status(), 'sources': json.loads((UPSTREAM / 'sources.lock.json').read_text())}

    @app.post('/upstream/task')
    def task(body: TaskRequest):
        return brain.run_text_task(body.text, body.confirmed, body.read_only_live)

    @app.post('/upstream/task/steps')
    def structured_task(body: StructuredTaskRequest):
        return brain.run_steps([s.model_dump() for s in body.steps], body.text, body.confirmed, body.read_only_live, body.constraint)

    @app.get('/upstream/skills')
    def skills(confirmed: bool = False, read_only_live: bool = False):
        return brain.capabilities(confirmed, read_only_live)

    @app.post('/upstream/task/{task_id}/cancel')
    def cancel(task_id: str):
        result = brain.cancel_task(task_id)
        if result['status'] == 'not_found':
            raise HTTPException(404, 'Task not found')
        return result

    @app.get('/upstream/task/{task_id}/events')
    def events(task_id: str):
        if brain.get_task(task_id) is None:
            raise HTTPException(404, 'Task not found')
        return brain.store.events(task_id)

    @app.post('/upstream/constraints/trace')
    def constraints(body: TraceRequest):
        try:
            return check_trace(body.formula, body.trace)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.post('/upstream/bumble/context')
    def context(body: TaskRequest):
        return brain.bumble_context(body.text)

    @app.post('/upstream/task/{task_id}/summary')
    def summary(task_id: str):
        task = brain.get_task(task_id)
        if task is None:
            raise HTTPException(404, 'Task not found')
        with brain.lock:
            return summarize(task, brain.runtime)

    return app
