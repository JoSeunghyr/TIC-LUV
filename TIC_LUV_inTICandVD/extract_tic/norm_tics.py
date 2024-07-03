import os
import torch
import numpy as np


data_path = r'.\small_tics'
data_path2 = r'.\small_tics_norm'
all_mats = os.listdir(data_path)

all_features = []
for f in all_mats:
    data = torch.load(os.path.join(data_path, f))
    all_features.append(np.array(data))
all_features = np.array(all_features).squeeze()

z_max = np.max(all_features, axis=0)
z_min = np.min(all_features, axis=0)

for f in all_mats:
    data = torch.load(os.path.join(data_path, f))
    tic = np.array(data).squeeze()
    features_norm = (tic - z_min) / (z_max - z_min)
    features_norm = torch.tensor(features_norm)
    torch.save(features_norm, os.path.join(data_path2, f))

print('Done')

