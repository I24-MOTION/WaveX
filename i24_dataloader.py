import torch
from torch.utils.data import Dataset
import numpy as np

class WavexDataset(Dataset):
    def __init__(self, dataset_rds_path, dataset_motion_path) -> None:
        super().__init__()
        print(torch.cuda.device_count()) #to activate the cuda on Lambda machine
        self.rds_data = np.load(dataset_rds_path, mmap_mode='r')
        self.motion_data = np.load(dataset_motion_path, mmap_mode='r')
        self.length = self.rds_data.shape[0]

        # Min-Max scaling parameters for rds
        self.rds_min = np.array([0, 0, 0])
        self.rds_max = np.array([100, 25, 100])

        # Min-Max scaling parameters for motion
        self.motion_min = 0
        self.motion_max = 100

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rds = self.rds_data[idx].astype(np.float32)
        motion = self.motion_data[idx].astype(np.float32)

        # Apply Min-Max scaling
        rds = (rds - self.rds_min[:, None, None]) / (self.rds_max[:, None, None] - self.rds_min[:, None, None])
        motion = (motion - self.motion_min) / (self.motion_max - self.motion_min)

        rds = torch.from_numpy(rds).type(torch.FloatTensor)
        motion = torch.from_numpy(motion).type(torch.FloatTensor)
        return rds, motion
