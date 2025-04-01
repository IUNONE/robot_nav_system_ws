import numpy as np
import torch
from typing import Dict

from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header

from .transform_utils import xyzr_to_pc2_msg
from .bbox_utils import boxes_to_corners_3d

"""
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
"""

LINES = [[0, 1], [1, 2], [2, 3], [3, 0], 
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]]

COLOR_MAPPING = {
    'car': [0., 1., 1.], 'Car' : [0., 1., 1.], 'truck': [0., 1., 1.], 'Vehicle': [0., 1., 1.], 'construction_vehicle': [0., 1., 1.], 'bus': [0., 1., 1.], 'trailer': [0., 1., 1.],
    'motorcycle': [0., 1., 0.], 'bicycle': [0., 1., 0.], 'Cyclist': [0., 1., 0.],
    'Pedestrian': [1., 1., 0.], 'pedestrian': [1., 1., 0.], 
    'barrier' : [1., 1., 1.], 'traffic_cone': [1., 1., 1.]
}

class VisulizeBBoxNode(Node):
    
    def __init__(self):
        super().__init__('vis_node')
        self.class_names = ['Vehicle', 'Pedestrian', 'Cyclist'] 
        
        self.pc_puber = self.create_publisher(PointCloud2, '/pointcloud', 10)
        # self.gt_bbox_puber = self.create_publisher(MarkerArray, '/detect_gtbox', 10)
        self.pred_bbox_puber = self.create_publisher(MarkerArray, '/detect_box3d', 10)
        self.pred_label_puber = self.create_publisher(MarkerArray, '/text_det', 10)

    def pub_bbox_msg(self, 
        pts, 
        pred_dicts: Dict=None,
        gt_boxes=None
    )-> int:
        
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'livox_frame'

        # 1. pts
        pointcloud_msg = xyzr_to_pc2_msg(pts, header)
        self.pc_puber.publish(pointcloud_msg)

        marker_bbox = MarkerArray()
        marker_bbox_text = MarkerArray()

        # 2. pred bbox
        if pred_dicts is not None:
            boxes = boxes_to_corners_3d(torch.from_numpy(pred_dicts['pred_boxes'])).to('cpu').numpy()
            score = pred_dicts['pred_scores']
            label = pred_dicts['pred_labels']

            marker_bbox.markers.clear()
            marker_bbox_text.markers.clear()

            for obid in range(boxes.shape[0]):
                marker = Marker()
                marker.header = header
                marker.id = obid * 2
                marker.action = Marker.ADD
                marker.type = Marker.LINE_LIST
                marker.lifetime = Duration(seconds=0).to_msg()

                color = COLOR_MAPPING[self.class_names[int(label[obid]) - 1]]
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = *color, 0.8
                marker.scale.x = 0.05

                marker.points = []
                for line in LINES:
                    ptu, ptv = boxes[obid][line[0]], boxes[obid][line[1]]
                    marker.points.append(Point(x=float(ptu[0]), y=float(ptu[1]), z=float(ptu[2])))
                    marker.points.append(Point(x=float(ptv[0]), y=float(ptv[1]), z=float(ptv[2])))
                marker_bbox.markers.append(marker)

                markert = Marker()
                markert.header = header
                markert.id = obid * 2 + 1
                markert.action = Marker.ADD
                markert.type = Marker.TEXT_VIEW_FACING
                markert.lifetime = Duration(seconds=0).to_msg()

                markert.color.r, markert.color.g, markert.color.b, markert.color.a = *color, 1.0
                markert.scale.z = 0.6
                markert.pose.orientation.w = 1.0

                markert.pose.position.x = (boxes[obid][0][0] + boxes[obid][2][0]) / 2
                markert.pose.position.y = (boxes[obid][0][1] + boxes[obid][2][1]) / 2
                markert.pose.position.z = (boxes[obid][0][2] + boxes[obid][4][2]) / 2
                markert.text = f"{self.class_names[label[obid]-1]}: {score[obid]:.2f}"
                marker_bbox_text.markers.append(markert)

            self.pred_bbox_puber.publish(marker_bbox)
            self.pred_label_puber.publish(marker_bbox_text)

            return boxes.shape[0]

        else:
            return 0
