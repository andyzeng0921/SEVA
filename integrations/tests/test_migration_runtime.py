from zeng_agent.config import AgentConfig
from upstream306.app import create_app


def test_destination_robot_has_separate_runtime(monkeypatch, tmp_path):
    runtime = tmp_path / 'upstream284'
    monkeypatch.setenv('ZENG_RUNTIME', str(runtime))
    config = AgentConfig(topic_node_id='0_284')
    app = create_app(config=config)
    brain = app.state.upstream_brain
    assert brain.runtime == runtime
    assert brain.status()['robot_topic_node_id'] == '0_284'
    assert brain.store.running() == []
    assert brain.memory.get_stats()['memory']['total'] == 0


def test_explicit_runtime_takes_precedence_over_environment(monkeypatch, tmp_path):
    monkeypatch.setenv('ZENG_RUNTIME', str(tmp_path / 'unrelated'))
    wanted = tmp_path / 'selected'
    app = create_app(config=AgentConfig(topic_node_id='0_284'), runtime=wanted)
    assert app.state.upstream_brain.runtime == wanted
