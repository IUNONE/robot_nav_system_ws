import argparse
from collections import defaultdict
import numpy as np
import torch
import copy
from typing import List, Dict
import sys

import rclpy
import numpy as np
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Pose
from nav_msgs.msg import MapMetaData, OccupancyGrid


class LocationNode(Node):

    def __init__(self, 
        model: torch.nn.Module, 
        args=None
    ):
        
        super().__init__()
        
        self.args = args
        self.model = model
        checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))  
        self.model.load_state_dict({k.replace('module.', ''):v for k, v in checkpoint['model_state_dict'].items()})
        self.model.cuda()
        self.model.eval()
        self.get_logger().info(f"Load {self.model.__class__.__name__} model from {args.ckpt}")

        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        # config
        self.offset_angle = 0
        self.offset_ground = 0.3
        self.point_cloud_range = [-40., -40., -1., 40., 40., 4.]   # [x_min, y_min, z_min, x_max, y_max, z_max]

        # suber
        self.pc_data_topic = '/livox/lidar'
        self.pc_data_suber = self.create_subscription(
            PointCloud2,
            self.pc_data_topic,
            self.online_inference,
            10
        )
        self.pc_data_suber
        self.get_logger().info(f"DetectionNode is ready to receive point cloud data from {self.pc_data_topic}")

    def set_init_pose(self, 
        init_pose: Pose,
    ):
        # TODO: global location alorit

        return 1
   
    def set_map_data(self, 
        map_data: MapData,
    ):
        self.map_data = map_data

    def get_current_pose(self):
        # data
        current_pose = self.model.get_current_pose()

        return current_pose

    def update_map_data(self, 
        map_data: MapData,
    ):
        self.map_data = map_data

def main(args=None):
    
    rclpy.init(args=sys.argv)
    
    model = LD_base()
    det_node = DetectionNode(model, args)
    rclpy.spin(det_node)

    det_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--ckpt', type=str, default='../ckpt/livox_model_1.pt', help='checkpoint to start from')
    args = parser.parse_args()

    main(args)
