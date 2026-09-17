"""
search.py — 搜索策略 Skill

实现消防桩搜索策略:
  1. 90° 环绕扫描 (0°/90°/180°/270°)
  2. 视觉检测每帧
  3. 发现目标 → 规划路径
  4. 未发现 → 选择方向探索
  5. 遇到障碍 → 退回起点的下一个方向
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .base import Skill, SkillInput, SkillOutput


class SearchPhase(str, Enum):
    """搜索阶段"""
    INIT = "init"                   # 初始化
    SURROUND_SCAN = "surround"      # 环绕扫描
    ANALYZE = "analyze"             # 分析检测结果
    NAVIGATE = "navigate"           # 前往目标
    EXPLORE = "explore"             # 探索方向
    OBSTACLE = "obstacle"           # 遇障处理
    RETURN = "return"               # 返回起点
    GREET = "greet"                 # 打招呼
    DONE = "done"                   # 完成
    FAILED = "failed"               # 失败


@dataclass
class ScanResult:
    """单帧扫描结果"""
    direction_label: str          # "N" / "E" / "S" / "W"
    heading_deg: float            # 绝对朝向 (°)
    fire_hydrant: bool
    direction: str                # 相对方位
    distance: str                 # near/mid/far
    has_person: bool
    confidence: float
    description: str


@dataclass
class SearchState:
    """搜索状态追踪"""
    phase: SearchPhase = SearchPhase.INIT
    current_heading: float = 0.0        # 当前朝向 (°)
    start_heading: float = 0.0          # 起始朝向
    scan_results: list[ScanResult] = field(default_factory=list)
    target_found: bool = False
    target_heading: float = 0.0         # 目标方向
    target_distance: str = "unknown"
    explored_directions: list[float] = field(default_factory=list)  # 已探索的绝对方向
    remaining_directions: list[float] = field(default_factory=list) # 待探索
    step_count: int = 0
    max_explore_steps: int = 10
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "phase": self.phase.value,
            "current_heading": round(self.current_heading, 1),
            "target_found": self.target_found,
            "target_heading": round(self.target_heading, 1),
            "target_distance": self.target_distance,
            "explored": [round(d, 1) for d in self.explored_directions],
            "remaining": [round(d, 1) for d in self.remaining_directions],
            "step_count": self.step_count,
        }


@dataclass
class SearchConfig:
    """搜索配置"""
    scan_angles: list[float] = field(default_factory=lambda: [0.0, 90.0, 180.0, 270.0])
    explore_distance: float = 1.5       # 探索前进距离 (m)
    max_explore_steps: int = 10         # 最大探索步数
    obstacle_backoff: float = 0.5       # 遇障后退距离 (m)
    min_confidence: float = 0.6         # 最小检测置信度
    dry_run: bool = True


class SearchSkill(Skill):
    """搜索策略技能 — 编排搜索流程

    依赖:
      - chassis (ChassisSkill)
      - visual_detect (VisualDetectSkill)
      - greet (GreetSkill)

    输入参数:
      - skill_registry: SkillRegistry 实例 (用于调用其他 skills)
      - max_steps: 最大探索步数

    输出:
      - success: bool
      - data: {phase, target_found, target_heading, scan_results, ...}
    """

    name = "search"
    description = "消防桩搜索策略: 环绕扫描→检测→导航→探索"
    version = "1.0.0"

    def __init__(self, config: Optional[SearchConfig] = None, registry: Optional[object] = None):
        self.config = config or SearchConfig()
        self.state = SearchState()
        self._registry = registry  # SkillRegistry 引用

    def set_registry(self, registry):
        """注入 SkillRegistry"""
        self._registry = registry

    def _call_skill(self, name: str, **params) -> SkillOutput:
        """调用已注册的 skill"""
        if self._registry is None:
            return SkillOutput(success=False, error=f"SkillRegistry 未注入, 无法调用 {name}")
        return self._registry.execute(name, SkillInput(params))

    # ── 阶段1: 初始化 ──

    def phase_init(self) -> SkillOutput:
        """初始化搜索: 重置状态, 记录起始朝向"""
        self.state = SearchState()
        self.state.phase = SearchPhase.SURROUND_SCAN
        self.state.current_heading = 0.0
        self.state.start_heading = 0.0
        self.state.remaining_directions = list(self.config.scan_angles)
        return SkillOutput(
            success=True,
            data=self.state.to_dict(),
            message="搜索初始化完成, 准备环绕扫描",
        )

    # ── 阶段2: 环绕扫描 ──

    def phase_surround_scan(self) -> SkillOutput:
        """90° 环绕扫描: 转向4个方向各拍一张"""
        scan_results = []
        directions_label = {0.0: "N(北)", 90.0: "E(东)", 180.0: "S(南)", 270.0: "W(西)"}

        for target_angle in self.config.scan_angles:
            # 转向目标角度
            if not self.config.dry_run:
                turn_result = self._call_skill(
                    "chassis", action="rotate_to", target_angle=target_angle,
                )
                if not turn_result.success:
                    return turn_result
                time.sleep(0.5)

            # 拍照检测
            detect_result = self._call_skill("visual_detect", save_image=True)
            if not detect_result.success:
                return detect_result

            data = detect_result.data
            scan_results.append(ScanResult(
                direction_label=directions_label.get(target_angle, f"{target_angle}°"),
                heading_deg=target_angle,
                fire_hydrant=data.get("fire_hydrant", "NO") == "YES",
                direction=data.get("direction", "unknown"),
                distance=data.get("distance", "unknown"),
                has_person=data.get("has_person", "NO") == "YES",
                confidence=float(data.get("confidence", 0)),
                description=data.get("description", ""),
            ))

            self.state.current_heading = target_angle

        self.state.scan_results = scan_results
        self.state.phase = SearchPhase.ANALYZE

        # 汇总
        found = [r for r in scan_results if r.fire_hydrant and r.confidence >= self.config.min_confidence]
        people = [r for r in scan_results if r.has_person]

        return SkillOutput(
            success=True,
            data={
                "scan_results": [
                    {
                        "heading": r.heading_deg,
                        "label": r.direction_label,
                        "fire_hydrant": r.fire_hydrant,
                        "direction": r.direction,
                        "distance": r.distance,
                        "has_person": r.has_person,
                        "confidence": r.confidence,
                    }
                    for r in scan_results
                ],
                "found_count": len(found),
                "people_count": len(people),
            },
            message=f"环绕扫描完成: 发现消防桩={len(found)}处, 看到人={len(people)}人",
        )

    # ── 阶段3: 分析 ──

    def phase_analyze(self) -> SkillOutput:
        """分析扫描结果, 决定下一步"""
        scan_results = self.state.scan_results

        # 按置信度排序, 找消防桩
        candidates = [r for r in scan_results if r.fire_hydrant and r.confidence >= self.config.min_confidence]
        candidates.sort(key=lambda r: r.confidence, reverse=True)

        if candidates:
            best = candidates[0]
            self.state.target_found = True
            self.state.target_heading = best.heading_deg
            self.state.target_distance = best.distance
            self.state.phase = SearchPhase.NAVIGATE
            return SkillOutput(
                success=True,
                data={
                    "decision": "navigate",
                    "target_heading": best.heading_deg,
                    "target_distance": best.distance,
                    "confidence": best.confidence,
                },
                message=f"发现消防桩! 方位={best.direction_label}, 距离={best.distance}, 置信度={best.confidence:.2f}",
            )

        # 检查是否有人, 打招呼
        if any(r.has_person for r in scan_results):
            self.state.phase = SearchPhase.GREET
            return SkillOutput(
                success=True,
                data={"decision": "greet"},
                message="未发现消防桩, 但看到人, 先打招呼",
            )

        # 未发现, 选择方向探索
        if self.state.remaining_directions:
            self.state.phase = SearchPhase.EXPLORE
            return SkillOutput(
                success=True,
                data={"decision": "explore", "remaining": self.state.remaining_directions},
                message="未发现消防桩, 准备探索",
            )

        # 所有方向都探索完毕
        self.state.phase = SearchPhase.FAILED
        return SkillOutput(
            success=False,
            data={"decision": "failed"},
            message="所有方向已搜索完毕, 未发现消防桩",
        )

    # ── 阶段4: 导航 ──

    def phase_navigate(self) -> SkillOutput:
        """前往消防桩目标"""
        target = self.state.target_heading
        distance = self.state.target_distance

        # 转向目标
        if not self.config.dry_run:
            turn_result = self._call_skill("chassis", action="rotate_to", target_angle=target)
            if not turn_result.success:
                return turn_result
            time.sleep(0.3)

        # 根据距离决定前进距离
        dist_map = {"near": 0.5, "mid": 1.0, "far": 2.0}
        move_distance = dist_map.get(distance, 1.0)

        if not self.config.dry_run:
            move_result = self._call_skill("chassis", action="forward", distance=move_distance)
            if not move_result.success:
                return move_result

        self.state.phase = SearchPhase.DONE
        return SkillOutput(
            success=True,
            data={"heading": target, "move_distance": move_distance},
            message=f"导航完成: 朝向{target}°前进{move_distance}m",
        )

    # ── 阶段5: 探索方向 ──

    def phase_explore(self) -> SkillOutput:
        """向一个方向探索"""
        if not self.state.remaining_directions:
            self.state.phase = SearchPhase.FAILED
            return SkillOutput(success=False, message="无剩余方向可探索")

        target_angle = self.state.remaining_directions.pop(0)
        self.state.explored_directions.append(target_angle)

        # 先转向
        if not self.config.dry_run:
            turn_result = self._call_skill("chassis", action="rotate_to", target_angle=target_angle)
            if not turn_result.success:
                return turn_result
            time.sleep(0.3)

        # 前进探索
        if not self.config.dry_run:
            move_result = self._call_skill("chassis", action="forward", distance=self.config.explore_distance)
            if not move_result.success:
                # 遇障 → 后退
                self._call_skill("chassis", action="backward", distance=self.config.obstacle_backoff)
                self.state.phase = SearchPhase.SURROUND_SCAN
                return SkillOutput(
                    success=True,
                    data={"action": "obstacle", "heading": target_angle},
                    message=f"方向{target_angle}°遇障, 退回准备重新扫描",
                )

        self.state.step_count += 1

        # 前进后拍照再看一次
        detect_result = self._call_skill("visual_detect", save_image=True)
        if detect_result.success:
            if detect_result.data.get("fire_hydrant") == "YES":
                self.state.target_found = True
                self.state.target_heading = target_angle
                self.state.target_distance = detect_result.data.get("distance", "mid")
                self.state.phase = SearchPhase.NAVIGATE
                return SkillOutput(
                    success=True,
                    data={"decision": "found_during_explore"},
                    message="探索中发现消防桩, 切换为导航",
                )

        if self.state.step_count >= self.config.max_explore_steps:
            # 返回起点, 换方向
            self.state.phase = SearchPhase.RETURN
            return SkillOutput(
                success=True,
                data={"decision": "return"},
                message=f"探索{self.state.step_count}步未发现, 返回起点换方向",
            )

        # 继续探索或重新扫描
        self.state.phase = SearchPhase.SURROUND_SCAN
        return SkillOutput(
            success=True,
            data={"step": self.state.step_count},
            message=f"探索第{self.state.step_count}步, 继续扫描",
        )

    # ── 阶段6: 返回 ──

    def phase_return(self) -> SkillOutput:
        """返回起始点"""
        # 转向起始方向
        if not self.config.dry_run:
            self._call_skill("chassis", action="rotate_to", target_angle=self.state.start_heading)
            time.sleep(0.3)
            # 后退 (简化: 假设起点在后方)
            self._call_skill("chassis", action="backward", distance=self.config.explore_distance * self.state.step_count)

        # 重置探索状态
        self.state.step_count = 0
        self.state.phase = SearchPhase.SURROUND_SCAN
        return SkillOutput(
            success=True,
            data={"heading": self.state.start_heading},
            message="已返回起点, 准备下一轮扫描",
        )

    # ── 阶段7: 打招呼 ──

    def phase_greet(self) -> SkillOutput:
        """向看到的人打招呼"""
        greet_result = self._call_skill("greet")
        self.state.phase = SearchPhase.EXPLORE if self.state.remaining_directions else SearchPhase.FAILED
        return SkillOutput(
            success=True,
            data={"greet": greet_result.data},
            message="已打招呼, 继续探索",
        )

    # ── 主执行入口 ──

    def execute(self, input: SkillInput) -> SkillOutput:
        """执行搜索 (外部通过 LangGraph 编排各阶段)"""
        # 此 skill 本身不直接运行完整搜索, 各阶段由 LangGraph 节点调用
        return SkillOutput(
            success=True,
            data={"phases": [p.value for p in SearchPhase]},
            message="搜索策略 skill 就绪, 请通过 LangGraph 节点调用各阶段",
        )

    def run_phase(self, phase: SearchPhase) -> SkillOutput:
        """按阶段名执行"""
        phase_map = {
            SearchPhase.INIT: self.phase_init,
            SearchPhase.SURROUND_SCAN: self.phase_surround_scan,
            SearchPhase.ANALYZE: self.phase_analyze,
            SearchPhase.NAVIGATE: self.phase_navigate,
            SearchPhase.EXPLORE: self.phase_explore,
            SearchPhase.RETURN: self.phase_return,
            SearchPhase.GREET: self.phase_greet,
            SearchPhase.DONE: lambda: SkillOutput(success=True, message="搜索完成"),
            SearchPhase.FAILED: lambda: SkillOutput(success=False, message="搜索失败"),
        }
        handler = phase_map.get(phase)
        if handler is None:
            return SkillOutput(success=False, error=f"未知阶段: {phase}")
        return handler()
