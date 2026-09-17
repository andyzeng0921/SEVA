"""Run original frontier + native Nav2 planner on a saved map, in DDS 224 only."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import yaml
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from lifecycle_msgs.srv import GetState
from tf2_ros import TransformBroadcaster

from .reward_policy import Experience, load_semantics, semantic_novelty, write_json
from .reward_runner import NativeCandidates, BRIDGE


def main():
    assert os.environ.get('ROS_DOMAIN_ID')=='224', 'This program must never publish on hardware DDS'
    parser=argparse.ArgumentParser();parser.add_argument('session',type=Path);args=parser.parse_args()
    s=args.session; folder=s/'reward/validation'
    metadata=json.loads((folder/'live-input-metadata.json').read_text())
    grid=np.load(folder/'live-input-grid.npz')['data']
    config=yaml.safe_load((s/'config/nav2.quality.yaml').read_text())
    test={key:copy.deepcopy(config[key]) for key in ['planner_server','global_costmap']}
    cp=test['global_costmap']['global_costmap']['ros__parameters']
    cp.update(use_sim_time=False,update_frequency=5.,publish_frequency=5.,transform_tolerance=.2)
    cp['static_layer'].update(map_topic='/map',first_map_only=False,subscribe_to_updates=False)
    test['planner_server']['ros__parameters']['use_sim_time']=False
    (folder/'isolated-planner.yaml').write_text(yaml.safe_dump(test,sort_keys=False))
    rclpy.init();node=rclpy.create_node('isolated_reward_validation')
    pub=node.create_publisher(OccupancyGrid,'/map',QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    broadcaster=TransformBroadcaster(node)
    msg=OccupancyGrid();msg.header.frame_id='map';msg.info.resolution=metadata['resolution']
    msg.info.height,msg.info.width=grid.shape;msg.info.origin.position.x,msg.info.origin.position.y=metadata['origin']
    msg.info.origin.orientation.w=1.;msg.data=grid.ravel().tolist()
    virtual_pose=metadata['pose'][:];last_publish=0.
    def pump(seconds):
        nonlocal last_publish
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if time.monotonic()-last_publish>.15:
                last_publish=time.monotonic();msg.header.stamp=node.get_clock().now().to_msg();pub.publish(msg)
                t=TransformStamped();t.header.frame_id='map';t.child_frame_id='base_link';t.header.stamp=msg.header.stamp
                t.transform.translation.x,t.transform.translation.y=virtual_pose[:2]
                t.transform.rotation.z,t.transform.rotation.w=math.sin(virtual_pose[2]/2),math.cos(virtual_pose[2]/2)
                broadcaster.sendTransform(t)
            rclpy.spin_once(node,timeout_sec=.02)
    def wait(future,seconds):
        end=time.monotonic()+seconds
        while not future.done() and time.monotonic()<end:pump(.03)
        if not future.done():raise RuntimeError('isolated_api_timeout')
        return future.result()
    log=(folder/'isolated-native.log').open('w')
    commands=[['ros2','run','nav2_planner','planner_server','--ros-args','--params-file',str(folder/'isolated-planner.yaml')],
        ['ros2','run','nav2_lifecycle_manager','lifecycle_manager','--ros-args','-r','__node:=isolated_planner_lifecycle',
         '-p','autostart:=true','-p',"node_names:=['planner_server']"], [BRIDGE]]
    processes=[subprocess.Popen(c,stdout=log,stderr=subprocess.STDOUT,start_new_session=True) for c in commands]
    results=[]
    try:
        status=node.create_client(GetState,'/planner_server/get_state')
        end=time.monotonic()+20
        while time.monotonic()<end:
            pump(.2)
            if status.service_is_ready() and wait(status.call_async(GetState.Request()),3).current_state.id==3:break
        else:raise RuntimeError('native_planner_failed_to_activate')
        pump(3)
        assert not node.get_publishers_info_by_topic('/topic_gv_target_cmd_vel_0_306')
        assert not any(name in ['controller_server','bt_navigator','behavior_server'] for name,_ in node.get_node_names_and_namespaces())
        native=NativeCandidates(node,wait);observations=load_semantics(s/'semantics')
        for label in ['actual_captured_pose','virtual_prior_graph_pose','synthetic_two_unknown_wings']:
            if label=='virtual_prior_graph_pose':
                # A second saved graph viewpoint is an offline replay, never a physical relocation.
                nodes=metadata['graph']['nodes'];xy=nodes.get('600') or nodes[str(sorted(map(int,nodes))[len(nodes)//2])]
                virtual_pose[:2]=xy[:2];pump(2)
            if label=='synthetic_two_unknown_wings':
                grid=np.full((400,400),100,np.int8)
                grid[20:380,20:190]=0
                grid[60:155,190:380]=-1
                grid[210:350,190:380]=-1
                metadata={**metadata,'resolution':.02,'origin':[0.,0.]}
                msg.info.resolution=.02;msg.info.height=400;msg.info.width=400
                msg.info.origin.position.x=0.;msg.info.origin.position.y=0.;msg.data=grid.ravel().tolist()
                virtual_pose[:]=[1.8,3.8,0.];pump(3)
            started=time.monotonic();snapshot=native.snapshot();checked=[]
            for c in snapshot['candidates']:
                c['semantic_novelty']=semantic_novelty(c,observations,metadata['graph'],metadata['map_epoch'])
                checked.append(native.validate(c))
            # A known occupied cell must not be selected, even with an enormous score hint.
            occupied=np.argwhere(grid==100)
            nearest=min(occupied,key=lambda rc:math.dist(
                [metadata['origin'][0]+(rc[1]+.5)*metadata['resolution'], metadata['origin'][1]+(rc[0]+.5)*metadata['resolution']],virtual_pose[:2]))
            xy=[metadata['origin'][0]+(nearest[1]+.5)*metadata['resolution'],metadata['origin'][1]+(nearest[0]+.5)*metadata['resolution']]
            unsafe=native.validate({'kind':'frontier','xy':xy,'yaw':0.,'visible_frontier_m':1e12,'dp_rank':1})
            assert not unsafe['path_valid'],unsafe
            ranked=Experience(metadata['map_epoch']).rank(checked+[unsafe],time.time())
            if label=='synthetic_two_unknown_wings':
                assert len(ranked)>=2,checked
                assert ranked[0]['cluster_cells']>ranked[-1]['cluster_cells'],ranked
                memory=Experience(metadata['map_epoch'])
                memory.record({'time':time.time(),'xy':ranked[0]['xy'],'success':False})
                alternatives=memory.rank(checked,time.time())
                assert alternatives and alternatives[0]['xy']!=ranked[0]['xy']
            row={'label':label,'virtual_pose':virtual_pose[:],'elapsed_s':time.monotonic()-started,
                'snapshot':snapshot,'checked':checked,'ranked':ranked,'occupied_goal_rejected':unsafe}
            results.append(row)
            print(json.dumps({'label':label,'frontiers':snapshot['native_frontier_count'],
                'pool':len(checked),'native_paths_valid':sum(c['path_valid'] for c in checked),'ranked':len(ranked),
                'selected':ranked[0] if ranked else None,'elapsed_s':row['elapsed_s']}),flush=True)
        write_json(folder/'native-reward-validation.json',{'hardware_connected':False,'motion_command_issued':False,
            'dds_domain':224,'input_origin':'saved live map; stamps refreshed only in isolated DDS',
            'native_core_unmodified':True,'results':results})
    finally:
        for p in processes:
            if p.poll() is None:os.killpg(p.pid,signal.SIGINT)
        for p in processes:
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=3)
        log.close();node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
