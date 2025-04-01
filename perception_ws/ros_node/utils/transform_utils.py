import torch
import numpy as np
from typing import List, Tuple, Dict
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct
import ros2_numpy

def xyzr_to_pc2_msg(
    pts: np.ndarray, 
    header: Header,
)-> PointCloud2:

    # data = {
    #     'xyz': pts[:, 0:3],
    #     'intensity': pts[:, 3]
    # }
    # pc2_msg = ros2_numpy.msgify(PointCloud2, data)
    # pc2_msg.header = header

    pts = pts.astype(np.float32)
    n_points = pts.shape[0]    
    data = np.zeros(n_points, dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32)])
    data['x'] = pts[:, 0]
    data['y'] = pts[:, 1]
    data['z'] = pts[:, 2]
    data['intensity'] = pts[:, 3]

    pc2_msg = PointCloud2()
    pc2_msg.header = header
    pc2_msg.height = 1
    pc2_msg.width = n_points
    pc2_msg.is_dense = True
    pc2_msg.is_bigendian = False
    pc2_msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
    ]
    pc2_msg.data = data.tobytes()

    return pc2_msg

def rotate_points_along_z(
    points: torch.Tensor,
    angle: torch.Tensor,
)-> torch.Tensor:
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    bs = points.shape[0]

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    
    zeros = angle.new_zeros(bs)
    ones = angle.new_ones(bs)

    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    
    return points_rot

def mask_points_out_of_range(
    pc: np.ndarray,
    valid_pc_range: List[float] 
)-> np.ndarray:
    
    valid_pc_range = np.array(valid_pc_range)
    valid_pc_range[3:6] -= 0.01  #np -> cuda .999999 = 1.0
    
    mask_x = (pc[:, 0] > valid_pc_range[0]) & (pc[:, 0] < valid_pc_range[3])
    mask_y = (pc[:, 1] > valid_pc_range[1]) & (pc[:, 1] < valid_pc_range[4])
    mask_z = (pc[:, 2] > valid_pc_range[2]) & (pc[:, 2] < valid_pc_range[5])
    mask = mask_x & mask_y & mask_z
    
    pc = pc[mask]
    
    return pc
