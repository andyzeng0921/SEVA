"""
skills — 模块化机器人技能组件

每个 skill 是独立的、可组合的机器人能力单元, 包含:
  - 明确的输入参数 (SkillInput)
  - 执行逻辑 (execute)
  - 结构化输出 (SkillOutput)

通过 SkillRegistry 注册和管理, 支持动态加载和 LangGraph 调用。
"""
from .base import Skill, SkillInput, SkillOutput, SkillRegistry as SkillReg
from .detect import VisualDetectSkill
from .chassis import ChassisSkill
from .search import SearchSkill
from .greet import GreetSkill

__all__ = [
    "Skill", "SkillInput", "SkillOutput", "SkillReg",
    "VisualDetectSkill", "ChassisSkill", "SearchSkill", "GreetSkill",
]
