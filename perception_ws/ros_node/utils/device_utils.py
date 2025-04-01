import torch
import numpy as np
from typing import Dict, List

def gpu2cpu(
    data_dict: Dict[str, torch.Tensor],
)-> Dict[str, np.ndarray]:
    
    for key in data_dict:
        if isinstance(data_dict[key], torch.Tensor):
            data_dict[key] = data_dict[key].cpu().numpy()
    
    torch.cuda.empty_cache()
    return data_dict

def cpu2gpu(
    batch_dict: Dict[str, np.ndarray],
)-> Dict[str, torch.Tensor]:
    
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()

    return batch_dict