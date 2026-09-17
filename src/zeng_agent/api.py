from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .service import RobotAgentService


class TextCommandRequest(BaseModel):
    text: str = Field(min_length=1)
    source: str = "text"
    confidence: float = 1.0


class BrainTextTaskRequest(BaseModel):
    text: str = Field(min_length=1)
    confirmed: bool = False


class OpenClawStartRequest(BaseModel):
    goal: str = Field(min_length=1)
    reset: bool = True


class OpenClawSessionRequest(BaseModel):
    session_id: str = Field(min_length=1)


class OpenClawTurnRequest(OpenClawSessionRequest):
    angle_deg: float


class OpenClawApproachRequest(OpenClawSessionRequest):
    distance_m: float = Field(gt=0)


class OpenClawStopRequest(BaseModel):
    session_id: str | None = None
    reason: str = "requested"


def create_app(service: RobotAgentService) -> FastAPI:
    app = FastAPI(title="zeng-agent", version="0.1.0")

    @app.get("/status")
    def get_status() -> dict:
        return service.status()

    @app.post("/command/text")
    def post_text_command(payload: TextCommandRequest) -> dict:
        return service.handle_text_command(payload.text, source=payload.source, confidence=payload.confidence)

    @app.post("/command/asr/local")
    def post_local_asr_command() -> dict:
        return service.handle_local_asr_command()

    @app.get("/brain/status")
    def get_brain_status() -> dict:
        return service.brain_status()

    @app.post("/brain/task/text")
    def post_brain_text_task(payload: BrainTextTaskRequest) -> dict:
        return service.handle_brain_text_task(payload.text, confirmed=payload.confirmed)

    @app.get("/brain/task/{task_id}")
    def get_brain_task(task_id: str) -> dict:
        return service.get_brain_task(task_id)

    @app.get("/openclaw/robot/status")
    def get_openclaw_robot_status() -> dict:
        return service.openclaw_status()

    @app.post("/openclaw/robot/session/start")
    def post_openclaw_start(payload: OpenClawStartRequest) -> dict:
        return service.openclaw_start(payload.goal, payload.reset)

    @app.post("/openclaw/robot/observe")
    def post_openclaw_observe(payload: OpenClawSessionRequest) -> dict:
        return service.openclaw_observe(payload.session_id)

    @app.post("/openclaw/robot/turn")
    def post_openclaw_turn(payload: OpenClawTurnRequest) -> dict:
        return service.openclaw_turn(payload.session_id, payload.angle_deg)

    @app.post("/openclaw/robot/approach")
    def post_openclaw_approach(payload: OpenClawApproachRequest) -> dict:
        return service.openclaw_approach(payload.session_id, payload.distance_m)

    @app.post("/openclaw/robot/stop")
    def post_openclaw_stop(payload: OpenClawStopRequest) -> dict:
        return service.openclaw_stop(payload.session_id, payload.reason)

    return app
