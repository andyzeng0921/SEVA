from __future__ import annotations

import json
from typing import Any

import requests

from .config import LlmConfig


SYSTEM_PROMPT = """你是一个机器人任务规划器。将用户的中文指令解析为结构化的机器人操作步骤。

可用的机器人技能列表：
1. navigate_to_waypoint - 导航到指定航点（参数: name=航点名称）
2. save_waypoint - 保存当前位置为航点（参数: name=航点名称）
3. list_waypoints - 列出所有航点（无参数）
4. robot_status - 查询机器人状态（无参数）
5. stop_navigation - 停止导航/急停（无参数）
6. arm_preset - 执行机械臂预设动作（参数: action_name=动作名称）
7. arm_stop - 停止机械臂运动（无参数）

要求：
- 用户可能一次说多个步骤（如"先去home，再挥手"），需要拆解为多个步骤
- 每个步骤必须映射到上述技能之一
- 如果无法理解用户指令，返回空的 steps 列表并给出错误信息
- 用 JSON 格式输出，不要包含 markdown 代码块标记

输出格式：
{
  "steps": [
    {
      "text": "用户原始指令片段",
      "intent": "技能名称",
      "slots": {"参数名": "参数值"},
      "explanation": "简要解释"
    }
  ],
  "message": "ok 或错误描述"
}"""


class LLMTaskPlanner:
    """通过兼容的聊天接口调用已配置的机器人任务规划模型。"""

    def __init__(
        self,
        config: LlmConfig,
        allowed_arm_presets: list[str] | None = None,
        waypoint_names: list[str] | None = None,
    ) -> None:
        self.config = config
        self._arm_presets = allowed_arm_presets or []
        self._waypoints = waypoint_names or []

    def _build_prompt(self) -> str:
        presets_str = ", ".join(self._arm_presets) if self._arm_presets else "reset, stop, wave"
        waypoints_str = ", ".join(self._waypoints) if self._waypoints else "（未指定）"
        return f"""你是一个机器人任务规划器。将用户的中文指令解析为结构化的机器人操作步骤。

可用的机器人技能列表：
1. navigate_to_waypoint - 导航到指定航点（参数: name=航点名称）
2. save_waypoint - 保存当前位置为航点（参数: name=航点名称）
3. list_waypoints - 列出所有航点（无参数）
4. robot_status - 查询机器人状态（无参数）
5. stop_navigation - 停止导航/急停（无参数）
6. arm_preset - 执行机械臂预设动作（参数: action_name=动作名称）
7. arm_stop - 停止机械臂运动（无参数）
8. arm_continuous_move - 连续移动机械臂（参数: direction=方向(up/down/left/right/forward/backward), amount_meters=移动距离(米), limb=手臂(left/right)）
9. arm_cross_wave - 一手上下一手交叉挥手（参数: phase=full_cycle 完整交叉，或 left_up/right_up 单侧），在身体前方安全执行
10. vision_navigate - 视觉导航前进（参数: goal_distance_m=目标距离(米)，默认2.0），通过前方摄像头逐步前进
11. vision_turn - 视觉转身（参数: direction=方向(left/right/auto), angle_deg=转身角度(度)，默认30），先视觉检测空间再执行转身
12. fire_search - 从头部前向共享内存相机搜索、对准并分段靠近消防栓（参数: instruction=用户原始要求）

已知的航点名：{waypoints_str}
已知的机械臂预设名：{presets_str}

要求：
- 用户可能一次说多个步骤（如"先去home，再挥手"），需要拆解为多个步骤
- 每个步骤必须映射到上述技能之一
- arm_preset 的 action_name 必须从预设列表中选取，如果用户说的动作不在预设中则返回 unsupported
- navigate_to_waypoint 的 name 必须是已知航点名
- 如果无法理解用户指令，返回空的 steps 列表并给出错误信息
- 用 JSON 格式输出，不要包含 markdown 代码块标记

输出格式：
{{
  "steps": [
    {{
      "text": "用户原始指令片段",
      "intent": "技能名称",
      "slots": {{"参数名": "参数值"}},
      "explanation": "简要解释"
    }}
  ],
  "message": "ok 或错误描述"
}}"""

    def plan(self, text: str) -> dict[str, Any]:
        """将用户文本指令规划为结构化步骤

        返回:
            {{"steps": [...], "message": "ok"}} 或
            {{"steps": [], "message": "错误信息"}}
        """
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self._build_prompt()},
                {"role": "user", "content": text},
            ],
            "temperature": 0.1,
            "max_tokens": 4096,
        }
        if self.config.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.config.enable_thinking}

        try:
            resp = requests.post(
                f"{self.config.base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.config.timeout_seconds,
            )
            resp.raise_for_status()
            body = resp.json()
            choice = body["choices"][0]
            if choice.get("finish_reason") == "length":
                return {"steps": [], "message": "LLM 输出被截断，未生成可执行计划"}
            msg = choice["message"]
            content = msg.get("content")
            if not content:
                return {"steps": [], "message": "LLM 返回空内容"}
            return self._parse_response(content.strip())
        except requests.Timeout:
            return {"steps": [], "message": "LLM 请求超时"}
        except requests.RequestException as e:
            return {"steps": [], "message": f"LLM 请求失败: {e}"}
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            return {"steps": [], "message": f"LLM 响应解析失败: {e}"}

    @staticmethod
    def _parse_response(content: str) -> dict[str, Any]:
        # 去掉可能的 markdown 代码块标记
        text = content.strip()

        # 1) 尝试从 ```json...``` 中提取
        json_match = __import__("re").search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        if json_match:
            text = json_match.group(1)

        # 2) 尝试找到最后一个完整的 JSON 对象
        if not json_match and text:
            # 找 { 和对应的 }
            brace_open = text.find("{")
            if brace_open >= 0:
                # 从最后一个 } 往前找配对
                depth = 0
                start = brace_open
                end = -1
                for i, ch in enumerate(text[brace_open:], brace_open):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                if end > start:
                    text = text[start:end]

        data = json.loads(text)
        steps = data.get("steps", [])
        message = data.get("message", "ok")
        return {"steps": steps, "message": message}
