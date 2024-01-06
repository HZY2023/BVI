import os
import numpy as np
import torch
import re
for index in range(0,114000):
    path = r'F:\clip-best'
        # path = os.path.join(path, 'train')
    path = sorted(os.listdir(path), key=lambda i: int(re.match(r'(\d+)', i).group()))
    src_img_features_item = path[index]
    src_img_features_item = os.path.join(r'F:\clip-best', path[index])
    img_npy = np.load(src_img_features_item, allow_pickle=True)
    img_tensor = torch.tensor(img_npy)
    img_tensor = img_tensor.to(torch.float32)
    src_img_features_item = img_tensor.view(img_tensor.size(0), img_tensor.size(1) * img_tensor.size(2), -1)