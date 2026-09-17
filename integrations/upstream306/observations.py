"""Read-only SHM2 transport adapter around the robot's existing frame reader."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np
from zeng_agent.models import ExecutionResult, utc_now_iso


def native_reader():
    directory = Path(__file__).resolve().parents[1] / 'vendor306'
    manifest = json.loads((directory / 'sources.json').read_text())
    for name, metadata in manifest.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != metadata['sha256']:
            raise RuntimeError('Native camera source differs from pinned robot version')
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    config = importlib.import_module('camera_config')
    reader = importlib.import_module('shm_camera')
    if Path(reader.__file__).resolve().parent != directory or Path(config.__file__).resolve().parent != directory:
        raise RuntimeError('Conflicting camera reader module')
    return config, reader


def capture_rgbd(output_dir, timeout=2.0, reader_pair=None):
    config, reader = reader_pair or native_reader()
    color_spec, depth_spec = (config.CAMERA_SPECS[name] for name in ['rgbd_head_color', 'rgbd_head_depth'])
    baseline = [reader.read_shm_metadata(spec) for spec in [color_spec, depth_spec]]
    deadline = time.monotonic() + timeout
    # Every invocation proves producer advancement, even if old SHM files remain.
    if any(m is None or len(m) < 9 for m in baseline):
        return ExecutionResult(False, 'SHM2 metadata unavailable', {'failure_code': 'OBSERVATION_UNAVAILABLE'})
    while time.monotonic() < deadline:
        metadata = [reader.read_shm_metadata(spec) for spec in [color_spec, depth_spec]]
        if all(m and len(m) >= 9 and m[8] > old[8] for m, old in zip(metadata, baseline)):
            if abs(metadata[0][8] - metadata[1][8]) <= 1:
                frames = [reader.read_shm_frame(spec, m) for spec, m in zip([color_spec, depth_spec], metadata)]
                if all(frame is not None for frame in frames):
                    epochs = [f.timestamp_ns / 1e9 for f in frames]
                    if all(reader.EPOCH_NS_MIN <= f.timestamp_ns <= reader.EPOCH_NS_MAX for f in frames):
                        if any(not -0.2 <= time.time() - stamp <= 0.5 for stamp in epochs):
                            time.sleep(.01)
                            continue
                        if abs(epochs[0] - epochs[1]) > .05:
                            time.sleep(.01)
                            continue
                    rgb = reader.frame_to_hwc(frames[0], False, rgb=True)
                    depth = reader.frame_to_hwc(frames[1], True)
                    if rgb.shape[:2] != depth.shape[:2]:
                        return ExecutionResult(False, 'RGB/depth dimensions differ', {'failure_code': 'OBSERVATION_INVALID'})
                    evidence_id = str(uuid.uuid4())
                    output = Path(output_dir).resolve() / f'{evidence_id}.npz'
                    output.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(output, rgb=rgb, depth_mm=depth)
                    return ExecutionResult(True, 'Fresh synchronized RGB-D pair captured', {
                        'mode': 'read_only_live', 'observation_id': evidence_id,
                        'captured_at': utc_now_iso(), 'producer_frame_ids': [m[8] for m in metadata],
                        'producer_timestamps_ns': [f.timestamp_ns for f in frames],
                        'freshness_basis': 'stable SHM2 copy and advancing producers',
                        'coordinate_frame': 'rgbd_head_color_optical_frame', 'position_validity': 'unavailable',
                        'world_pose_available': False, 'metric_target_positions': None,
                        'validity': 'observation_only; invalidate action parameters after robot/camera/target change',
                        'artifact': str(output), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                        'shape': list(rgb.shape), 'depth_unit': 'millimetres'})
        time.sleep(.01)
    return ExecutionResult(False, 'No fresh synchronized RGB-D pair before deadline', {'failure_code': 'OBSERVATION_STALE'})
