"""Render saved native maps and validated camera viewpoints; never controls ROS."""
import argparse
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('session', type=Path)
p.add_argument('--status', choices=['running', 'parked'], required=True)
a = p.parse_args()
s = a.session.resolve()

def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default

meta = yaml.safe_load((s/'maps/current.yaml').read_text())
grid = np.asarray(Image.open(s/'maps'/meta['image']))
res = float(meta['resolution'])
records = sorted([read(f) for f in (s/'semantics').glob('*.json')], key=lambda x: x['captured_at'])
views, rejected = [], []
for r in records:
    if not r.get('schema_valid'):
        rejected.append({'id': r['observation_id'], 'reason': r.get('validation_error', 'schema invalid')})
        continue
    tf = r['viewpoint']
    assert tf['header']['frame_id'] == 'map'
    t, q = tf['transform']['translation'], tf['transform']['rotation']
    yaw = math.atan2(2*(q['w']*q['z']+q['x']*q['y']), 1-2*(q['y']**2+q['z']**2))
    views.append({'id': r['observation_id'], 'captured_at': r['captured_at'],
                  'map_position_m': [t['x'], t['y']], 'base_heading_rad': yaw,
                  'model': r['model'], 'scene': r['semantic_scene'],
                  'graph_anchor': r.get('graph_anchor'),
                  'image': 'observations/'+r['observation_id']+'.jpg'})

index = {'coordinate_frame': 'map', 'occupancy_map': 'maps/current.yaml',
         'spatial_semantics': 'Robot viewpoint at capture time, not measured object world coordinates. Historical poses have not been reoptimized after later graph corrections.',
         'confidence_semantics': 'Uncalibrated model scores; schema validation does not establish recognition accuracy.',
         'viewpoints': views, 'rejected_observations': rejected}
(s/'semantic-viewpoints.json').write_text(json.dumps(index, ensure_ascii=False, indent=2))

runs = sorted((s/'evidence').glob('coverage-*'))
runs = [r for r in runs if r.is_dir() and (r/'events.jsonl').exists()]
latest = runs[-1] if runs else s/'evidence/no-run'
result = read(latest/'result.json', {})
live = read(latest/'live.json', {})
straight_records = [r for r in (s/'evidence').glob('straight-*') if r.is_dir() and (r/'native-straight-result.json').exists() and (r/'result.json').exists()]
final_action = {}
if straight_records:
    straight = max(straight_records, key=lambda r: (r/'result.json').stat().st_mtime)
    action, settled = read(straight/'native-straight-result.json'), read(straight/'result.json')
    final_action = {'source': str(straight.relative_to(s)), 'action': 'native DriveOnHeading',
                    'requested_distance_m': action['requested_distance_m'],
                    'requested_speed_m_s': action['requested_speed_m_s'],
                    'status': action.get('status'), 'error_code': action.get('error_code'),
                    'distance_through_stop_m': settled['distance_m'],
                    'parked_verified': settled['parked_verified'],
                    'later_than_latest_coverage': (straight/'result.json').stat().st_mtime > ((latest/'result.json').stat().st_mtime if (latest/'result.json').exists() else float('inf'))}
completed_runs = []
for r in runs:
    data = read(r/'result.json', {})
    if data:
        completed_runs.append({'id': r.name, 'distance_m': data.get('distance_m', 0.),
                               'reason': data.get('reason'), 'parked_verified': data.get('parked_verified'),
                               'strategy': 'reward' if any(e.get('event', '').startswith('reward_') for e in data.get('events', [])) else 'legacy',
                               'loop_closure_verified': data.get('loop_closure_verified', False)})
undocks = [read(f) for f in (s/'departures').glob('*/evidence/undock-result.json')]
alignment = read(s/'evidence/upstream-live-alignment.json', {})
revisit_alignment = read(s/'evidence/after-revisit-map-consistency.json', {})
graph_audit = read(s/'evidence/native-posegraph-revisit-constraints.json', {})
closure_replay = read(s/'evidence/native-closure-replay/result.json', {})
clearances = [dict(read(s/'evidence'/name, {}), source='evidence/'+name) for name in
              ['resume-parked-clearance.json', 'straight-request-clearance.json',
               'door-center-final-clearance.json', 'cleared-resume-cad.json',
               'cleared-latest-cad.json', 'continuation-final-cad.json', 'reward-resume-final-cad.json']]
clearance = max(clearances, key=lambda value: value.get('time', 0))
nearest = (clearance.get('nearest_returns') or [None])[0]
inside_stop_margin = bool(nearest and nearest['cad_distance'] < .075)
clearance_is_recent = 0 <= dt.datetime.now().timestamp()-clearance.get('time', 0) < 30
current_block = {'reason': 'measured_clearance_below_stop_margin' if inside_stop_margin and clearance_is_recent and a.status == 'parked' else None,
                 'checked_at': clearance.get('time'), 'nearest_lidar_return': nearest,
                 'source': clearance.get('source'), 'recent_measurement': clearance_is_recent,
                 'required_body_stop_margin_m': 0.075,
                 'motion_stopped': a.status == 'parked',
                 'physical_intervention_pending': inside_stop_margin and clearance_is_recent and a.status == 'parked'}
if a.status == 'parked' and final_action.get('later_than_latest_coverage') and final_action.get('error_code') == 723:
    current_block.update(reason='native_collision_prediction_blocked_motion',
                         physical_intervention_pending=True,
                         native_action_source=final_action['source'])
latest_reward_action = next((e.get('outcome') for e in reversed(result.get('events', []))
                            if e.get('event') == 'reward_goal_interrupted'), None)
controller_log = (latest/'native-navigation.log').read_text() if (latest/'native-navigation.log').exists() else ''
if (a.status == 'parked' and latest_reward_action
        and 'RegulatedPurePursuitController detected collision ahead' in controller_log):
    current_block.update(reason='native_controller_collision_prediction', physical_intervention_pending=True,
                         native_action_source=str(latest.relative_to(s)),
                         native_error_code=latest_reward_action.get('native_error_code'))
free, occupied, unknown = [int(np.count_nonzero(grid == v)) for v in [254, 0, 205]]
quality = {'robot': 306, 'session': s.name, 'exported_at': dt.datetime.now(dt.timezone.utc).isoformat(),
           'status': a.status, 'map_resolution_m': res, 'width': grid.shape[1], 'height': grid.shape[0],
           'observed_free_m2': free*res**2, 'free_cells': free, 'occupied_cells': occupied,
           'unknown_cells': unknown, 'latest_run': latest.name,
           'run_result': {k: v for k, v in result.items() if k not in ['trajectory', 'events', 'graphs']},
           'latest_live_progress': live, 'valid_semantic_viewpoints': len(views),
           'completed_runs': completed_runs,
           'recorded_coverage_odometry_distance_m': sum(r['distance_m'] for r in completed_runs) + (live.get('distance_m', 0.) if not result else 0.),
           'recorded_reward_strategy_distance_m': sum(r['distance_m'] for r in completed_runs if r['strategy'] == 'reward'),
           'native_departures': [{k:v for k,v in d.items() if k not in ['trajectory','physical_commands']} for d in undocks],
           'rejected_semantic_viewpoints': len(rejected),
           'loop_closure_verified': any(r['loop_closure_verified'] for r in completed_runs),
           'coverage_complete': False, 'environment_coverage_fraction': None,
           'independent_metric_accuracy_validated': False,
           'earlier_static_scan_consistency': alignment,
           'after_revisit_static_scan_consistency': revisit_alignment,
           'native_posegraph_revisit_audit': {k:v for k,v in graph_audit.items() if k != 'nonlocal_revisit_constraints'},
           'isolated_native_closure_replay': closure_replay,
           'current_motion_block': current_block,
           'final_native_motion_action': final_action,
           'latest_reward_action': latest_reward_action,
           'native_map_provenance': read(s/'evidence/current-map-provenance.json', {}),
           'door_center_configuration': read(s/'evidence/door-center-candidate.json', {}),
           'monitor_regression': read(s/'evidence/monitor-regression.json', {}),
           'monitor_live_validation': read(s/'evidence/latest-feedback-live-soak.json', {}),
           'reward_exploration_upgrade': read(s/'reward/upstream-lock.json', {}),
           'limitations': ['Resolution is not independently verified accuracy.',
                           'Unknown cells remain unknown; full room/corridor coverage is not demonstrated.',
                           'Semantic labels belong to camera viewpoints; object world coordinates are unverified.',
                           'Scan/map consistency at one resting pose does not certify the later entire map.',
                           'Nonlocal graph edges demonstrate existing revisit constraints; their creation pathway is not identified by this read-only audit.',
                           'The isolated replay did not observe a dedicated global loop-closure event. The unmodified live node does not expose a count of those events.']}
(s/'quality-report.json').write_text(json.dumps(quality, ensure_ascii=False, indent=2))

x0, y0, origin_yaw = meta['origin']
assert abs(origin_yaw) < 1e-8, 'Renderer requires an axis-aligned native map'
h, w = grid.shape
fig, ax = plt.subplots(figsize=(12, 9), layout='constrained')
ax.imshow(grid, cmap='gray', vmin=0, vmax=255, origin='upper',
          extent=[x0, x0+w*res, y0, y0+h*res], interpolation='nearest')
for i, v in enumerate(views, 1):
    x, y = v['map_position_m']; heading = v['base_heading_rad']
    ax.scatter([x], [y], c='#087da6', s=32, edgecolors='white', zorder=4)
    ax.arrow(x, y, .3*math.cos(heading), .3*math.sin(heading), width=.008,
             head_width=.09, color='#087da6', length_includes_head=True, zorder=3)
    angle = i*2.39996
    # Spread labels for repeat visits to the same anchor; map coordinates stay unchanged.
    repeat_offsets = {3: (3, 30), 4: (30, -34), 5: (45, 15), 6: (0, 53),
                      9: (-48, 12), 12: (-45, -28)}
    label_offset = repeat_offsets.get(i, (22*math.cos(angle), 22*math.sin(angle)))
    ax.annotate(f'P{i}', (x, y), xytext=label_offset,
                textcoords='offset points', fontsize=9, color='#00516e',
                arrowprops={'arrowstyle':'-', 'lw':.5, 'color':'#087da6'},
                bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':.9, 'pad':1.5}, zorder=5)
ax.set_title(f'306 | Lidar map with Qwen3.5 camera viewpoints\n'
             f'{res*100:.0f} cm grid | {free*res**2:.1f} m² observed free space | coverage incomplete',
             loc='left', fontsize=15, pad=15)
ax.text(.99, .99, 'White: observed free\nBlack: occupied\nGray: unknown\nP: camera viewpoint\nArrow: base heading',
        ha='right', va='top', transform=ax.transAxes, fontsize=10,
        bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':.9, 'pad':6})
ax.set_xlabel('map x (m)'); ax.set_ylabel('map y (m)'); ax.set_aspect('equal')
fig.savefig(s/'maps/semantic-map.png', dpi=160, bbox_inches='tight')
fig.savefig(s/'maps/semantic-map.svg', bbox_inches='tight')
plt.close(fig)

state_cn = '正在监督运行' if a.status == 'running' else '本段运行已停止，停车证据见结果记录'
blocking_note = state_cn + '。'
if current_block['reason'] == 'native_collision_prediction_blocked_motion':
    blocking_note = f'最近一次原生低速直行目标为 {final_action["requested_distance_m"]:.2f} 米，实际至停车累计约 {final_action["distance_through_stop_m"]:.3f} 米，随后原生碰撞预测返回 COLLISION_AHEAD。转向和后退恢复也受阻，底盘已停车。'
elif current_block['reason'] == 'native_controller_collision_prediction':
    blocking_note = f'最近一次奖励回访目标被原生导航接受，但 RPP 控制器检测到前方碰撞风险后中止，错误码 {latest_reward_action.get("native_error_code")}；本次实际里程 {result.get("distance_m", 0):.3f} 米，底盘已停车。'
elif current_block['physical_intervention_pending']:
    blocking_note = '当前近距回波进入完整机身外轮廓的停止余量，底盘已停车，等待现场清理。'
lines = ['# 306 雷达与语义探索记录', '', f'2026-09-14；{state_cn}。', '',
         f'当前保存地图为 {w} × {h} 栅格，分辨率 {res:.2f} 米，已观测空闲面积 {free*res**2:.2f} 平方米。', '',
         f'Qwen3.5 共保存 {len(views)} 条通过结构校验的语义观察记录（含重复回访），{len(rejected)} 个未通过校验的记录已排除。标签绑定拍摄时刻的机器人位置，尚未验证物体世界坐标，也未随后续图优化重新优化历史观察点。', '',
         '地图覆盖尚未完成。2 厘米为栅格分辨率，不代表已验证的定位精度；返航成功或图节点增加也不能单独证明闭环。', '',
         f'本会话已记录探索里程 {quality["recorded_coverage_odometry_distance_m"]:.2f} 米（包含原地转动时的里程计微小漂移；脱桩另记，人工开出不计入自主探索里程）。{blocking_note}', '',
         f'上次原生位姿图审计记录了 {graph_audit.get("native_graph_nodes", 0)} 个节点、{graph_audit.get("native_graph_edges", 0)} 条边。该次只读检查找到 {graph_audit.get("nonlocal_revisit_constraint_count", 0)} 条跨次回访约束：两个端点之间曾行驶至少 5 米，优化后端点距离不超过 1 米；筛选条件和原始端点保存在证据 JSON 中。', '',
         '这些约束支持“已进行回访并建立图约束”，还不能证明全部环境闭环完成。隔离实录回放未观测到专用全局闭环事件，原版在线节点也没有暴露该事件计数。', '',
         '一次停车回访后的 545 个有效激光端点中，90.83% 距已有占据栅格不超过 10 厘米；这是该时刻的局部一致性指标。', '',
         '## 保存结果', '',
         '- [语义地图 PNG](maps/semantic-map.png) / [SVG](maps/semantic-map.svg)',
         '- [原生导航地图](maps/current.yaml) / [PGM](maps/current.pgm)',
         '- [原生 SLAM 图](maps/current.posegraph) / [续建数据](maps/current.data)',
         '- [语义索引](semantic-viewpoints.json) / [质量记录](quality-report.json)',
         f'- [原生回访约束](evidence/native-posegraph-revisit-constraints.json) / [最近一次机身净距测量]({clearance.get("source")})；测量只描述记录时刻。',
         '- observations：原始 RGB-D、照片、拍摄时间与 TF；semantics：模型输出和结构校验结果。',
         '- bags：原始雷达、里程计、TF 和运行记录；大体积原始包保留在机器人会话目录。',
         '- evidence/upstream-rotation-replay：隔离 ROS Domain 216 的实录回放验证，不回放底盘命令。', '',
         '## 语义观察点', '', '| 点 | 拍摄时 map 坐标（米） | 模型识别内容 | 照片 |', '|---|---|---|---|']
for i, v in enumerate(views, 1):
    labels = '、'.join(dict.fromkeys(o['label'] for o in v['scene'].get('objects', [])))
    x, y = v['map_position_m']
    lines.append(f'| P{i} | {x:.2f}, {y:.2f} | {labels} | [查看]({v["image"]}) |')
lines += ['', '## 迁移与配置', '',
          '运动、避碰、规划、建图和前沿选择使用原生实现；会话代码负责调用、监测和保存。', '',
          '- [SLAM Toolbox 2.8.5 完整源码](https://github.com/SteveMacenski/slam_toolbox/tree/02afdde003313a10b8d21461a92d9e5f4f0bc5f2)：独立编译，启用精确平移与转角扫描筛选，源码未改写。',
          '- [m-explore-ros2](https://github.com/robo-friends/m-explore-ros2/tree/326cf8a0b487c34246bb8f3326afbcd69576dc60)：原版前沿探索器。',
          '- [Nav2 1.3.10](https://github.com/ros-navigation/navigation2/tree/1.3.10)：原生脱桩、Smac2D、RPP、速度平滑与 Collision Monitor。',
          '- 两个雷达安装坐标复用当前厂家 robot_v2_2 配置，通过原生 tf2 静态发布器独立运行。',
          '- 原版探索器读取原始占据图选择前沿，实际导航继续使用膨胀代价图。隔离测试证据见 evidence/native-frontier-comparison.json。',
          '- 前沿导航使用 Nav2 自带 navigate_w_replanning_time.xml；无法规划的目标由原版探索器加入黑名单。返访使用原生恢复树。',
          '- 当前完整机身 CAD 凸轮廓外保留至少 7.5 厘米停止余量；门洞居中配置将导航足迹扩大到停止轮廓外再留约 2 厘米。近障限速 0.04 米/秒，并启用两秒足迹预测避碰。无执行器输出的原生避碰测试见 evidence/collision-contour-probe.json，测试位置与当前停车位置不同，不能替代当前位置净距检查。',
          '- 门洞居中参数已完成隔离规划比较及实机参数读回；实际过门效果仍待验证，详见 [后续直行与配置记录](validation/door-center/README.md)。',
          '- 足迹包含当前双臂外形；视觉标签不会清除雷达障碍或绕过避碰。',
          '- 已按用户授权调用另一套后台自带的 stop-backend.sh 暂停控制及其重启进程。', '']
lines += ['## 复现与续建', '',
          '- coverage_session.py 负责原版探索器、Nav2、原生地图保存与语义采集之间的衔接；实际运动仍须通过当前避碰链。',
          '- maps/current.posegraph 与 maps/current.data 为同一停车检查点的原生序列化文件；完整历史检查点位于 maps/coverage-*。',
          '- validation/closure-observer 保留原生闭环监听与只读图检查的源码，只允许在隔离 ROS Domain 220 运行。',
          '- build_semantic_report.py 生成此报告与索引；export_session.py 导出结果并生成 SHA-256 清单。',
          '- 导出包排除 API 密钥和运行凭据。原始 rosbag、已编译程序及完整上游仓库保留在机器人上。', '']
if (s/'reward/upstream-lock.json').exists():
    lines += ['## 奖励策略升级', '',
              '当前启动入口已默认切换为开源 WFD/MRTSP/DP 候选加奖励与经验排序；旧流程保留为 legacy。累计里程包含两种策略，逐段记录见质量 JSON。', '',
              f'奖励策略实机运行已记录 {quality["recorded_reward_strategy_distance_m"]:.3f} 米里程；包含原地转动时的微小里程计漂移，人工或其他控制程序的移动不计入。已完成历史点回访，但覆盖率与闭环数量的改善尚未得到对照验证。详细评分、源码版本和验证结果见 [奖励探索接入说明](reward/README.md)。', '',
              '新语义索引保留 graph_anchor 字段，供目标排序读取当前 SLAM 节点附近的观察覆盖度；本图仍绘制拍摄时刻的原始观察位置。', '']
(s/'README.md').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps({k: quality[k] for k in ['observed_free_m2', 'valid_semantic_viewpoints', 'latest_run', 'status']}, ensure_ascii=False))
