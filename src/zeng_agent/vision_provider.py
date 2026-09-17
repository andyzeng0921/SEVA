"""Camera capture and multimodal vision access.

All named shared-memory cameras are read from SHM.  A camera name is never
silently converted to the rear V4L2 fisheye camera.
"""
from __future__ import annotations

import base64
import hashlib
import os
import struct
import time
from dataclasses import dataclass, field, replace
from typing import Optional

import cv2
import requests


CAMERA_MAP = {
    "rear_fisheye": {
        "name": "身后鱼眼",
        "type": "v4l2",
        "index": 8,
        "description": "Global Shutter Camera, rear-facing fisheye",
    },
    "rear_fisheye_alt": {
        "name": "身后鱼眼(备用)",
        "type": "v4l2",
        "index": 7,
        "description": "Backup rear-facing fisheye",
    },
    "front_left": {
        "name": "前方左目",
        "type": "shm",
        "shm_path": "/dev/shm/camera_image_buffer_head_left_jpeg",
        "description": "Head front-left color camera",
    },
    "front_right": {
        "name": "前方右目",
        "type": "shm",
        "shm_path": "/dev/shm/camera_image_buffer_head_right_jpeg",
        "description": "Head front-right color camera",
    },
    "hand_left": {
        "name": "左手相机",
        "type": "shm",
        "shm_path": "/dev/shm/camera_image_buffer_hand_left_jpeg",
        "description": "Left-hand camera; not guaranteed to face forward",
    },
    "hand_right": {
        "name": "右手相机",
        "type": "shm",
        "shm_path": "/dev/shm/camera_image_buffer_hand_right_jpeg",
        "description": "Right-hand camera; not guaranteed to face forward",
    },
    "rgbd_color": {
        "name": "头部RGBD彩色相机",
        "type": "shm",
        "shm_path": "/dev/shm/camera_image_buffer_rgbd_head_color",
        "meta_path": "/dev/shm/camera_metadata_struct_rgbd_head_color",
        "encoding": "raw_bgr",
        "description": "Head RGBD color stream",
    },
}

SHM_PATHS = {
    name: info["shm_path"]
    for name, info in CAMERA_MAP.items()
    if info["type"] == "shm"
}
SHM_PATHS["rgbd_depth"] = "/dev/shm/camera_image_buffer_rgbd_head_depth"


@dataclass
class VisionConfig:
    api_url: str = "http://127.0.0.1:8000/v1/chat/completions"
    api_key: str = ""
    model: str = "qwen3.5-35b-a3b"
    camera_source: str = "front_left"
    camera_index: int = 8
    camera_width: int = 640
    camera_height: int = 480
    max_tokens: int = 500
    timeout: int = 60
    warmup_frames: int = 5
    fresh_frame_timeout: float = 2.0
    fresh_frame_poll_interval: float = 0.08
    enable_thinking: bool | None = None


@dataclass(frozen=True)
class FrameMetadata:
    source: str
    transport: str
    frame_id: str
    captured_at: float
    byte_length: int
    producer_frame_id: int | None = None


@dataclass
class VisionResult:
    success: bool
    content: str = ""
    reasoning: str = ""
    usage: dict = field(default_factory=dict)
    error: str = ""


class VisionProvider:
    def __init__(self, config: VisionConfig) -> None:
        self.config = config

    def _check_api(self) -> VisionResult:
        try:
            response = requests.post(
                self.config.api_url,
                json={
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 10,
                },
                headers=self._headers(),
                timeout=10,
            )
            if response.status_code == 200:
                return VisionResult(success=True, content="API 连接正常")
            return VisionResult(success=False, error=f"API status={response.status_code}: {response.text[:200]}")
        except Exception as exc:
            return VisionResult(success=False, error=str(exc))

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

    def _capture_shm(self, shm_path: str) -> tuple[Optional[bytes], Optional[str]]:
        fd: int | None = None
        try:
            fd = os.open(shm_path, os.O_RDONLY)
            size = os.fstat(fd).st_size
            full = os.read(fd, size)
        except Exception as exc:
            return None, f"SHM 读取失败 {shm_path}: {exc}"
        finally:
            if fd is not None:
                os.close(fd)

        start = full.find(b"\xff\xd8")
        end = full.rfind(b"\xff\xd9")
        if start < 0 or end < start:
            return None, f"SHM 中未找到完整 JPEG ({shm_path})"
        return full[start:end + 2], None

    def _capture_v4l2(self, index: int) -> tuple[Optional[bytes], Optional[str]]:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        if not cap.isOpened():
            return None, f"无法打开 V4L2 摄像头 index={index}"
        try:
            for _ in range(self.config.warmup_frames):
                cap.read()
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok:
            return None, f"V4L2 拍照失败 index={index}"
        encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not encoded:
            return None, "JPEG 编码失败"
        return buffer.tobytes(), None

    def _capture_raw_bgr(self, data_path: str, meta_path: str) -> tuple[Optional[bytes], int | None, Optional[str]]:
        """Read one stable raw BGR frame using the producer metadata contract."""
        metadata_format = "<QIIIIQI"
        metadata_size = struct.calcsize(metadata_format)
        for _ in range(3):
            try:
                with open(meta_path, "rb", buffering=0) as meta_file:
                    before = meta_file.read(metadata_size)
                frame_id, width, height, channels, depth_bytes, data_size, _ = struct.unpack(
                    metadata_format, before
                )
                expected = width * height * channels * depth_bytes
                if frame_id <= 0 or width <= 0 or height <= 0 or channels != 3 or depth_bytes != 1:
                    return None, None, f"无效RGB元数据: frame={frame_id}, {width}x{height}x{channels}"
                if data_size != expected or data_size > 64 * 1024 * 1024:
                    return None, None, f"RGB数据长度异常: metadata={data_size}, expected={expected}"
                with open(data_path, "rb", buffering=0) as data_file:
                    raw = data_file.read(data_size)
                with open(meta_path, "rb", buffering=0) as meta_file:
                    after = meta_file.read(metadata_size)
                after_frame_id = struct.unpack(metadata_format, after)[0]
                if frame_id != after_frame_id or len(raw) != data_size:
                    continue
                frame = memoryview(raw)
                import numpy as np

                image = np.frombuffer(frame, dtype=np.uint8).reshape((height, width, channels))
                encoded, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if not encoded:
                    return None, None, "RGBD 彩色帧 JPEG 编码失败"
                return buffer.tobytes(), int(frame_id), None
            except (OSError, ValueError, struct.error) as exc:
                return None, None, f"RGBD 共享内存读取失败: {exc}"
        return None, None, "RGBD 帧在读取期间持续变化，未取得一致快照"

    def _candidate_sources(self) -> list[str]:
        source = self.config.camera_source
        if source == "auto_front":
            return ["rgbd_color", "front_left", "front_right"]
        return [source]

    def _capture_source(self, source: str) -> tuple[Optional[bytes], str, int | None, Optional[str]]:
        info = CAMERA_MAP.get(source)
        if info is None:
            return None, "unknown", None, f"未知摄像头源: {source}"
        if info["type"] == "shm":
            path = info["shm_path"]
            if not os.path.exists(path):
                return None, "shm", None, f"SHM 路径不存在: {path}"
            if info.get("encoding") == "raw_bgr":
                meta_path = str(info["meta_path"])
                if not os.path.exists(meta_path):
                    return None, "shm_raw", None, f"SHM 元数据路径不存在: {meta_path}"
                data, producer_id, error = self._capture_raw_bgr(path, meta_path)
                return data, "shm_raw", producer_id, error
            data, error = self._capture_shm(path)
            return data, "shm", None, error
        data, error = self._capture_v4l2(int(info["index"]))
        return data, "v4l2", None, error

    def capture_frame_with_meta(
        self,
        require_newer_than: str | None = None,
        wait_timeout: float | None = None,
    ) -> tuple[Optional[bytes], Optional[FrameMetadata], Optional[str]]:
        """Capture a frame and optionally require bytes different from a prior frame.

        SHM files are commonly updated through mmap, so filesystem mtime is not a
        reliable freshness signal.  The JPEG digest is the safety token used after
        every robot motion.
        """
        timeout = self.config.fresh_frame_timeout if wait_timeout is None else max(0.0, wait_timeout)
        deadline = time.monotonic() + timeout
        errors: list[str] = []
        while True:
            errors.clear()
            for source in self._candidate_sources():
                data, transport, producer_id, error = self._capture_source(source)
                if data is None:
                    errors.append(error or f"{source} capture failed")
                    continue
                frame_id = f"{source}:{producer_id}" if producer_id is not None else hashlib.sha256(data).hexdigest()
                if require_newer_than and frame_id == require_newer_than:
                    errors.append(f"{source} 仍是上一帧")
                    continue
                metadata = FrameMetadata(
                    source=source,
                    transport=transport,
                    frame_id=frame_id,
                    captured_at=time.time(),
                    byte_length=len(data),
                    producer_frame_id=producer_id,
                )
                return data, metadata, None
            if time.monotonic() >= deadline:
                detail = "; ".join(errors) or "没有可用摄像头"
                return None, None, f"未获得新的前向图像: {detail}"
            time.sleep(self.config.fresh_frame_poll_interval)

    def capture_live_frame(
        self,
        require_newer_than: str | None = None,
    ) -> tuple[Optional[bytes], Optional[FrameMetadata], Optional[str]]:
        """Return a frame only after proving that its producer is updating.

        For the first frame there is no prior motion token, so each candidate
        camera is sampled once and then polled for a different JPEG from that
        same camera.  This prevents an old but readable SHM snapshot from
        authorizing the first robot turn.
        """
        if require_newer_than:
            return self.capture_frame_with_meta(require_newer_than=require_newer_than)

        errors: list[str] = []
        for source in self._candidate_sources():
            locked = VisionProvider(replace(self.config, camera_source=source))
            _, baseline, error = locked.capture_frame_with_meta(wait_timeout=0.0)
            if baseline is None:
                errors.append(error or f"{source} 无法读取")
                continue
            data, metadata, error = locked.capture_frame_with_meta(
                require_newer_than=baseline.frame_id,
                wait_timeout=self.config.fresh_frame_timeout,
            )
            if data is not None and metadata is not None:
                return data, metadata, None
            errors.append(error or f"{source} 未更新")
        return None, None, "前向相机流未更新: " + "; ".join(errors)

    def capture_frame(self) -> tuple[Optional[bytes], Optional[str]]:
        data, _, error = self.capture_frame_with_meta(wait_timeout=0.0)
        return data, error

    def describe_image(self, image_bytes: bytes, prompt: str = "请描述这张图片") -> VisionResult:
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "model": self.config.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                ],
            }],
            "max_tokens": self.config.max_tokens,
        }
        thinking = self.config.enable_thinking
        if thinking is None and self.config.model.lower().startswith("qwen3"):
            thinking = False
        if thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": thinking}
        try:
            response = requests.post(
                self.config.api_url,
                json=payload,
                headers=self._headers(),
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]
        except Exception as exc:
            return VisionResult(success=False, error=f"视觉 API 失败: {exc}")
        content = message.get("content") or ""
        if choice.get("finish_reason") == "length" or not content:
            return VisionResult(success=False, content=content, usage=data.get("usage", {}),
                                error="视觉输出被截断或缺少最终内容")
        return VisionResult(
            success=True,
            content=content,
            reasoning=message.get("reasoning") or "",
            usage=data.get("usage", {}),
        )

    def capture_and_describe(self, prompt: str = "请描述这张图片") -> VisionResult:
        image, error = self.capture_frame()
        if image is None:
            return VisionResult(success=False, error=f"拍照: {error}")
        return self.describe_image(image, prompt)


def create_vision_provider(api_key: str, camera_source: str = "front_left") -> VisionProvider:
    return VisionProvider(VisionConfig(api_key=api_key, camera_source=camera_source))
