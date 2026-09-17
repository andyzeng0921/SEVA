from __future__ import annotations

import re

from .arm_motion_planner import parse_amount, parse_direction, parse_limb
from .models import ParsedIntent


class IntentParser:
    _navigate_patterns = [
        re.compile(r"^(去|前往|导航到)\s*(?P<name>[\w\-]+)$", re.IGNORECASE),
    ]
    _save_patterns = [
        re.compile(r"^(保存点位|记录点位|记住这里)\s*(?P<name>[\w\-]+)$", re.IGNORECASE),
    ]
    _status_patterns = [
        re.compile(r"^(查询状态|机器人状态|当前状态)$", re.IGNORECASE),
    ]
    _list_patterns = [
        re.compile(r"^(查询点位|查看点位|有哪些点位)$", re.IGNORECASE),
    ]
    _stop_patterns = [
        re.compile(r"^(停止|急停|停下|停止导航)$", re.IGNORECASE),
    ]
    _arm_reset_patterns = [
        re.compile(r"^(?:复位|归位|回家)\s*(?:机械臂|手臂)?$", re.IGNORECASE),
        re.compile(r"^(?:机械臂|手臂)\s*(?:复位|归位|回家)$", re.IGNORECASE),
    ]
    _arm_stop_patterns = [
        re.compile(r"^(机械臂停止|停止机械臂)$", re.IGNORECASE),
    ]
    _arm_continuous_patterns = [
        re.compile(
            r"^(?:机械臂|手臂)?"
            r"(?:左臂|右臂|手臂)?"
            r"\s*"
            r"(?P<direction>向下|向上|向左|向右|向前|向后|往上|往下|往左|往右|往前|往后)"
            r"\s*"
            r"(?:移动|运动|走)?"
            r"\s*"
            r"(?P<amount>一点点|一点|点儿|稍微|少许|一些|一下|中等|一些些|很多|大量|最大)?"
            r"\s*"
            r"(?:移动|运动|走)?"
            r"\s*"
            r"$",
            re.IGNORECASE,
        ),
    ]

    _arm_cross_wave_patterns = [
        re.compile(
            r"^(?:做)?\s*"
            r"(?:一手上下一手交叉|上下挥手|交叉挥手|两手上下交叉)"
            r"\s*$",
            re.IGNORECASE,
        ),
    ]

    _vision_nav_patterns = [
        re.compile(
            r"^(?:视觉)?\s*"
            r"(?:导航(?:前进)?|前进|往前走|直行)"
            r"\s*(?P<distance>\d+(?:\.\d+)?)?\s*(?:米|m)?\s*$",
            re.IGNORECASE,
        ),
    ]

    _fire_search_patterns = [
        re.compile(
            r"^(?:请)?\s*(?:寻找|搜索|查找|去找)\s*(?:并靠近|并前往|并导航到)?\s*(?:一下)?\s*(?:消防栓|消防桩)$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^(?:请)?\s*(?:找到|寻找|搜索|查找)\s*(?:消防栓|消防桩)\s*(?:并前往|并靠近|并导航到)?$",
            re.IGNORECASE,
        ),
    ]

    _vision_turn_patterns = [
        re.compile(
            r"^(?:视觉|安全)?\s*"
            r"(?P<direction>左转|右转|向左转|向右转|转身|向后转|掉头)"
            r"\s*(?P<angle>\d+(?:\.\d+)?)?\s*(?:度|°)?\s*$",
            re.IGNORECASE,
        ),
    ]

    def parse(self, text: str) -> ParsedIntent:
        cleaned = self._normalize(text)

        for pattern in self._navigate_patterns:
            if match := pattern.match(cleaned):
                return ParsedIntent(intent="navigate_to_waypoint", slots={"name": match.group("name")}, confidence=0.95)

        for pattern in self._save_patterns:
            if match := pattern.match(cleaned):
                return ParsedIntent(intent="save_waypoint", slots={"name": match.group("name")}, confidence=0.95)

        if any(pattern.match(cleaned) for pattern in self._list_patterns):
            return ParsedIntent(intent="list_waypoints", confidence=0.95)

        if any(pattern.match(cleaned) for pattern in self._status_patterns):
            return ParsedIntent(intent="robot_status", confidence=0.95)

        if any(pattern.match(cleaned) for pattern in self._stop_patterns):
            return ParsedIntent(intent="stop_navigation", confidence=0.98)

        if any(pattern.match(cleaned) for pattern in self._arm_reset_patterns):
            return ParsedIntent(intent="arm_preset", slots={"action_name": "reset"}, confidence=0.95)

        if any(pattern.match(cleaned) for pattern in self._arm_stop_patterns):
            return ParsedIntent(intent="arm_stop", confidence=0.98)

        if any(pattern.match(cleaned) for pattern in self._arm_cross_wave_patterns):
            return ParsedIntent(intent="arm_cross_wave", slots={"phase": "full_cycle"}, confidence=0.95)

        if any(pattern.match(cleaned) for pattern in self._fire_search_patterns):
            return ParsedIntent(
                intent="fire_search",
                slots={"instruction": cleaned},
                confidence=0.98,
            )

        for pattern in self._vision_nav_patterns:
            m = pattern.match(cleaned)
            if m:
                distance = m.group("distance")
                return ParsedIntent(
                    intent="vision_navigate",
                    slots={"goal_distance_m": float(distance) if distance else 2.0},
                    confidence=0.92,
                )

        for pattern in self._vision_turn_patterns:
            m = pattern.match(cleaned)
            if m:
                dir_raw = m.group("direction")
                # 映射方向
                if "左" in dir_raw or "逆" in dir_raw:
                    direction = "left"
                    angle_val = m.group("angle")
                    angle = float(angle_val) if angle_val else 30.0
                elif "右" in dir_raw:
                    direction = "right"
                    angle_val = m.group("angle")
                    angle = float(angle_val) if angle_val else 30.0
                elif "后" in dir_raw or "掉头" in dir_raw:
                    direction = "left"
                    angle = 180.0
                else:
                    direction = "left"
                    angle_val = m.group("angle")
                    angle = float(angle_val) if angle_val else 30.0
                return ParsedIntent(
                    intent="vision_turn",
                    slots={"direction": direction, "angle_deg": angle},
                    confidence=0.92,
                )

        for pattern in self._arm_continuous_patterns:
            if match := pattern.match(cleaned):
                direction = parse_direction(match.group("direction")) or "down"
                amount = parse_amount(match.group("amount") or "")
                limb = parse_limb(match.group(0))
                return ParsedIntent(
                    intent="arm_continuous_move",
                    slots={
                        "direction": direction,
                        "amount_meters": amount,
                        "limb": limb,
                    },
                    confidence=0.85,
                )

        if "机械臂" in cleaned:
            return ParsedIntent(intent="unsupported", slots={"raw_text": cleaned}, confidence=0.4, requires_confirmation=True)

        return ParsedIntent(intent="unsupported", slots={"raw_text": cleaned}, confidence=0.2, requires_confirmation=True)

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())
