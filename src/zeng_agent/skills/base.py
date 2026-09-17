"""
base.py — Skill 基类和 SkillRegistry

每个 Skill 都是独立的能力单元:
  - 输入: SkillInput (params dict)
  - 输出: SkillOutput (success, data, error)
  - 元数据: name, description, version
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SkillInput:
    """Skill 输入参数"""
    params: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)


@dataclass
class SkillOutput:
    """Skill 执行输出"""
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "message": self.message,
        }


class Skill(ABC):
    """技能基类 — 所有 skill 的抽象父类

    子类必须实现:
      - name (str): 技能名称
      - description (str): 技能描述
      - execute(input) -> SkillOutput: 执行逻辑
    """

    name: str = ""
    description: str = ""
    version: str = "1.0.0"

    @abstractmethod
    def execute(self, input: SkillInput) -> SkillOutput:
        """执行技能, 返回结构化输出"""
        ...

    def validate(self, input: SkillInput) -> bool:
        """参数校验 (子类可覆写)"""
        return True

    def safe_execute(self, input: SkillInput) -> SkillOutput:
        """带错误处理的安全执行"""
        try:
            if not self.validate(input):
                return SkillOutput(
                    success=False,
                    error=f"参数校验失败: {input.params}",
                    message=f"{self.name} 参数校验失败",
                )
            return self.execute(input)
        except Exception as e:
            logger.exception(f"[{self.name}] 执行异常: {e}")
            return SkillOutput(
                success=False,
                error=str(e),
                message=f"{self.name} 执行异常",
            )

    def __repr__(self) -> str:
        return f"Skill({self.name} v{self.version})"


class SkillRegistry:
    """技能注册中心 — 管理所有 skill 实例

    用法:
        registry = SkillRegistry()
        registry.register(VisualDetectSkill())
        skill = registry.get("visual_detect")
        result = skill.safe_execute(SkillInput({"camera": "front_left"}))
    """

    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """注册一个 skill"""
        if skill.name in self._skills:
            logger.warning(f"覆盖已注册的 skill: {skill.name}")
        self._skills[skill.name] = skill
        logger.info(f"注册 skill: {skill.name} v{skill.version}")

    def register_many(self, skills: list[Skill]) -> None:
        """批量注册 skills"""
        for s in skills:
            self.register(s)

    def get(self, name: str) -> Skill | None:
        """按名称获取 skill"""
        return self._skills.get(name)

    def execute(self, name: str, input: SkillInput | None = None) -> SkillOutput:
        """按名称执行 skill (带错误处理)"""
        skill = self.get(name)
        if skill is None:
            return SkillOutput(
                success=False,
                error=f"未找到 skill: {name}",
                message=f"Skill '{name}' 不存在",
            )
        return skill.safe_execute(input or SkillInput())

    def list_skills(self) -> list[dict[str, str]]:
        """列出所有已注册的 skill 元数据"""
        return [
            {"name": s.name, "description": s.description, "version": s.version}
            for s in self._skills.values()
        ]

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __len__(self) -> int:
        return len(self._skills)
