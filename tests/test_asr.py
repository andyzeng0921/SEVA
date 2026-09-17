import unittest
import pathlib
import sys
from subprocess import CompletedProcess
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from zeng_agent.asr import LocalMicASRProvider, RemoteASRProvider
from zeng_agent.models import CommandText


class AsrProviderTests(unittest.TestCase):
    def test_local_provider_requires_transcriber_command(self):
        provider = LocalMicASRProvider(device="default", sample_rate=16000, transcriber_command=None)

        result = provider.transcribe_existing_file("/tmp/fake.wav")

        self.assertFalse(result.success)
        self.assertIn("transcriber command", result.message.lower())

    def test_remote_provider_normalizes_response(self):
        provider = RemoteASRProvider(base_url="http://example.com")

        payload = provider.parse_response({"text": "去 home", "confidence": 0.88})

        self.assertIsInstance(payload, CommandText)
        self.assertEqual(payload.text, "去 home")
        self.assertEqual(payload.confidence, 0.88)

    def test_local_provider_parses_json_transcriber_output(self):
        provider = LocalMicASRProvider(device="default", sample_rate=16000, transcriber_command="fixture-transcriber {wav_path}")
        output = CompletedProcess(args=[], returncode=0,
                                  stdout='{"text": "机械臂复位", "confidence": 0.93}', stderr="")
        with patch("zeng_agent.asr.subprocess.run", return_value=output):
            result = provider.transcribe_existing_file("/tmp/fake.wav")

        self.assertTrue(result.success)
        self.assertEqual(result.command.text, "机械臂复位")
        self.assertEqual(result.command.confidence, 0.93)


if __name__ == "__main__":
    unittest.main()
