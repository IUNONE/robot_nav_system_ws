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
import ros2_numpy

from livoxdetection.models.ld_base_v1 import LD_base
from utils.vis_node import VisulizeBBoxNode
from utils.transform_utils import mask_points_out_of_range
from utils.device_utils import gpu2cpu, cpu2gpu

class DetectionNode(VisulizeBBoxNode):

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

    def online_inference(self, 
        msg: PointCloud2,
    ):
        # data
        pc: Dict[np.ndarray] = ros2_numpy.numpify(msg)
        N = pc['intensity'].shape[0]
        pts = np.zeros((N, 4))
        pts[:, :3] = copy.deepcopy(np.float32(pc['xyz']))
        pts[:, 3] = copy.deepcopy(np.float32(pc['intensity']).reshape(-1))
        pts[:, 2] += pts[:, 0] * np.tan(self.offset_angle / 180. * np.pi) + self.offset_ground
        rviz_points = copy.deepcopy(pts)
        pts = mask_points_out_of_range(pts, self.point_cloud_range)

        # inference
        data_dict = {'points': pts}
        batched_data = self.collate_batch([data_dict])
        batched_data = cpu2gpu(batched_data)
        with torch.no_grad(): 
            torch.cuda.synchronize()
            self.starter.record()
            pred_dicts = self.model.forward(batched_data)
            self.ender.record()
            torch.cuda.synchronize()
            curr_latency = self.starter.elapsed_time(self.ender)
        self.get_logger().info(f'inference time (ms): {curr_latency}')
        
        # visualization
        pred_dicts = gpu2cpu(pred_dicts[0])
        n_detections = self.pub_bbox_msg(rviz_points[:, 0:4], pred_dicts)
        self.get_logger().info(f'number of detections: {n_detections}')

    @staticmethod
    def collate_batch(
        sample_list: List[Dict[str, np.ndarray]],
    )-> Dict[str, np.ndarray]:
        """
            collate list of dicts(one sample is a dict) to dict of batched arrays

            'points': [N*bs, ch+1] with batch index at first column
            'batch_size': bs
        """
        # collate each value for each key, list of dicts to dict of lists
        data_dict = defaultdict(list)
        for sample in sample_list:
            for key, value in sample.items():
                data_dict[key].append(value)

        bs = len(data_dict)
        ret = {}

        # dict of lists to dict of batched arrays
        for key, value_list in data_dict.items():
            if key == 'points':
                coords = []
                for i_sample, coord in enumerate(value_list):
                    # [N, ch] -> [N, ch+1] add batch index
                    coord_pad = np.pad(coord, ((0, 0), (1, 0)), mode='constant', constant_values=i_sample)
                    coords.append(coord_pad)
                # -> [N * bs, ch+1]
                ret[key] = np.concatenate(coords, axis=0)
            else:
                raise NotImplementedError(f"Unsupported key: {key}")
        ret['batch_size'] = bs
        return ret

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
