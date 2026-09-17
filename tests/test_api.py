import unittest
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from zeng_agent.api import create_app
from zeng_agent.config import AgentConfig
from zeng_agent.service import RobotAgentService


class ApiTests(unittest.TestCase):
    def test_status_endpoint(self):
        app = create_app(
            RobotAgentService(
                config=AgentConfig(),
                waypoint_names=["home"],
            )
        )
        client = TestClient(app)

        response = client.get("/status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["components"]["nlu"], "ok")

    def test_text_command_endpoint(self):
        app = create_app(
            RobotAgentService(
                config=AgentConfig(),
                waypoint_names=["home"],
            )
        )
        client = TestClient(app)

        response = client.post("/command/text", json={"text": "去 home"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["parsed"]["intent"], "navigate_to_waypoint")
        self.assertTrue(payload["result"]["success"])

    def test_local_asr_command_endpoint(self):
        class FakeAsrProvider:
            def capture_and_transcribe(self):
                from zeng_agent.models import AsrResult, CommandText

                return AsrResult(
                    success=True,
                    message="ok",
                    command=CommandText(text="机械臂复位", source="local_asr", confidence=0.91),
                )

        app = create_app(
            RobotAgentService(
                config=AgentConfig(),
                waypoint_names=["home"],
                local_asr_provider=FakeAsrProvider(),
            )
        )
        client = TestClient(app)

        response = client.post("/command/asr/local")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["command"]["source"], "local_asr")
        self.assertEqual(payload["parsed"]["intent"], "arm_preset")
        self.assertTrue(payload["result"]["success"])

    def test_brain_status_endpoint(self):
        app = create_app(
            RobotAgentService(
                config=AgentConfig(),
                waypoint_names=["home"],
            )
        )
        client = TestClient(app)

        response = client.get("/brain/status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["enabled"])

    def test_brain_text_task_requires_confirmation(self):
        from zeng_agent.brain import BrainOrchestrator
        from zeng_agent.executor import AgentExecutor
        from zeng_agent.nlu import IntentParser

        config = AgentConfig(waypoint_names=["home"])
        executor = AgentExecutor(config=config, waypoint_names=config.waypoint_names)
        app = create_app(
            RobotAgentService(
                config=config,
                waypoint_names=["home"],
                executor=executor,
                brain_orchestrator=BrainOrchestrator(config=config, parser=IntentParser(), executor=executor),
            )
        )
        client = TestClient(app)

        response = client.post("/brain/task/text", json={"text": "\u53bb home"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "blocked")

    def test_confirmed_brain_text_task_executes(self):
        from zeng_agent.brain import BrainOrchestrator
        from zeng_agent.executor import AgentExecutor
        from zeng_agent.nlu import IntentParser

        config = AgentConfig(waypoint_names=["home"])
        executor = AgentExecutor(config=config, waypoint_names=config.waypoint_names)
        app = create_app(
            RobotAgentService(
                config=config,
                waypoint_names=["home"],
                executor=executor,
                brain_orchestrator=BrainOrchestrator(config=config, parser=IntentParser(), executor=executor),
            )
        )
        client = TestClient(app)

        response = client.post("/brain/task/text", json={"text": "\u53bb home", "confirmed": True})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "completed")

    def test_brain_task_can_be_fetched(self):
        from zeng_agent.brain import BrainOrchestrator
        from zeng_agent.executor import AgentExecutor
        from zeng_agent.nlu import IntentParser

        config = AgentConfig(waypoint_names=["home"])
        executor = AgentExecutor(config=config, waypoint_names=config.waypoint_names)
        app = create_app(
            RobotAgentService(
                config=config,
                waypoint_names=["home"],
                executor=executor,
                brain_orchestrator=BrainOrchestrator(config=config, parser=IntentParser(), executor=executor),
            )
        )
        client = TestClient(app)
        created = client.post("/brain/task/text", json={"text": "\u67e5\u8be2\u72b6\u6001"}).json()

        response = client.get(f"/brain/task/{created['task_id']}")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["found"])
        self.assertEqual(payload["task_id"], created["task_id"])


if __name__ == "__main__":
    unittest.main()
