from __future__ import annotations

import shlex


def build_whispercpp_transcriber_command(
    python_bin: str,
    helper_script: str,
    whisper_bin: str,
    model_path: str,
    language: str = "zh",
) -> str:
    return (
        f"{shlex.quote(python_bin)} {shlex.quote(helper_script)} "
        f"--binary {shlex.quote(whisper_bin)} "
        f"--model {shlex.quote(model_path)} "
        f"--language {shlex.quote(language)} "
        f"--wav {{wav_path}}"
    )


def build_arm_preset_commands(
    python_bin: str,
    helper_script: str,
    topic_node_id: str,
    presets: list[str],
) -> dict[str, str]:
    commands: dict[str, str] = {
        "reset": (
            f"{shlex.quote(python_bin)} {shlex.quote(helper_script)} "
            f"reset --topic-node-id {shlex.quote(topic_node_id)}"
        ),
        "stop": (
            f"{shlex.quote(python_bin)} {shlex.quote(helper_script)} "
            f"reset --topic-node-id {shlex.quote(topic_node_id)}"
        ),
    }
    for preset in presets:
        commands[preset] = (
            f"{shlex.quote(python_bin)} {shlex.quote(helper_script)} "
            f"play --topic-node-id {shlex.quote(topic_node_id)} "
            f"--action-name {shlex.quote(preset)}"
        )
    return commands
