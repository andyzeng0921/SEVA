"""Read-only hardware snapshot for isolated exploration/planning validation."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
from nav_msgs.msg import OccupancyGrid, Odometry
from visualization_msgs.msg import MarkerArray
from tf2_ros import Buffer, TransformListener

from .reward_policy import write_json


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('session',type=Path); args=parser.parse_args()
    import sys
    sys.path.insert(0,str(args.session))
    from session_health import graph_summary, angle
    rclpy.init(); node=rclpy.create_node('zeng306_readonly_reward_inputs')
    buffer=Buffer(); listener=TransformListener(buffer,node)
    state={}; odometry=[]
    node.create_subscription(OccupancyGrid,'/map',lambda m:state.update(map=m),
        QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    node.create_subscription(MarkerArray,'/slam_toolbox/graph_visualization',lambda m:state.update(graph=m),1)
    node.create_subscription(Odometry,'/topic_gv_wheel_odom_0_306',lambda m:odometry.append(m),qos_profile_sensor_data)
    until=time.monotonic()+5
    while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.025)
    grid=state['map']; tf=buffer.lookup_transform('map','base_link',rclpy.time.Time())
    owners=node.get_publishers_info_by_topic('/map'); assert len(owners)==1
    assert len(odometry)>40 and max(abs(v) for m in odometry[-40:] for v in
        [m.twist.twist.linear.x,m.twist.twist.linear.y,m.twist.twist.angular.z]) < .003
    output=args.session/'reward/validation'; output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'live-input-grid.npz',data=np.asarray(grid.data,dtype=np.int8).reshape(grid.info.height,grid.info.width))
    metadata={'captured_at':time.time(),'original_map_stamp':rclpy.time.Time.from_msg(grid.header.stamp).nanoseconds/1e9,
        'resolution':grid.info.resolution,'origin':[grid.info.origin.position.x,grid.info.origin.position.y],
        'pose':[tf.transform.translation.x,tf.transform.translation.y,angle(tf.transform.rotation)],
        'map_epoch':bytes(owners[0].endpoint_gid).hex(),'graph':graph_summary(state['graph']),
        'motion_command_issued':False,'stationary_odom_samples':len(odometry),'source':'306 live ROS, read only'}
    write_json(output/'live-input-metadata.json',metadata)
    print(json.dumps({k:v for k,v in metadata.items() if k!='graph'}))
    node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
