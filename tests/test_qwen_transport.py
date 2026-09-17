"""Incomplete model replies must never become executable robot plans."""
from unittest.mock import Mock, patch

import pytest

from zeng_agent.config import LlmConfig
from zeng_agent.llm_planner import LLMTaskPlanner
from zeng_agent.vision_provider import VisionConfig, VisionProvider


@pytest.mark.parametrize('content,reason', [
    ('{"steps":[{"intent":"arm_preset","slots":{"action_name":"wave"}}]}', 'length'),
    ('', 'stop'),
])
def test_incomplete_reply_cannot_authorize_steps(content, reason):
    response = Mock()
    response.json.return_value = {'choices': [{'finish_reason': reason, 'message': {
        'content': content, 'reasoning': '{"steps":[{"intent":"arm_preset"}]}'}}]}
    planner = LLMTaskPlanner(LlmConfig(model='qwen3-32b', enable_thinking=False))
    with patch('zeng_agent.llm_planner.requests.post', return_value=response):
        assert planner.plan('测试')['steps'] == []


def test_image_transport_uses_multimodal_model_and_final_json():
    response = Mock()
    response.json.return_value = {'choices': [{'finish_reason': 'stop', 'message': {'content': '{"objects":[]}'}}]}
    provider = VisionProvider(VisionConfig(model='qwen3.5-35b-a3b'))
    with patch('zeng_agent.vision_provider.requests.post', return_value=response) as post:
        result = provider.describe_image(b'image-for-transport-test')
        payload = post.call_args.kwargs['json']
        assert result.success
        assert payload['model'] == 'qwen3.5-35b-a3b'
        assert payload['chat_template_kwargs'] == {'enable_thinking': False}
        assert payload['messages'][0]['content'][1]['type'] == 'image_url'


def test_truncated_semantics_are_retained_but_not_validated():
    response = Mock()
    response.json.return_value = {'choices': [{'finish_reason': 'length', 'message': {'content': '{"objects":['}}]}
    with patch('zeng_agent.vision_provider.requests.post', return_value=response):
        result = VisionProvider(VisionConfig()).describe_image(b'image-for-transport-test')
    assert not result.success
    assert result.content == '{"objects":['


@pytest.mark.parametrize('box', [[0, 0, 1001, 500], [20, 20, 10, 100]])
def test_invalid_image_boxes_are_not_map_annotations(box):
    from upstream306.semantic_map import SemanticObject
    with pytest.raises(ValueError):
        SemanticObject(label='object', bbox_xyxy_1000=box, confidence=0.8)
