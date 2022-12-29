
import os
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset

class CANDataset(Dataset):
    def __init__(self, root_dir, is_binary=False, transform=None):
        self.root_dir = Path(root_dir)
        self.is_binary = is_binary
        # self.is_train = is_train
        self.transform = transform
        self.total_size = len(os.listdir(self.root_dir))
            
    def __getitem__(self, idx):
        filename = f'{idx}.npz'
        filename = self.root_dir / filename
        data = np.load(filename)
        X, y = data['X'], data['y']
        if self.is_binary and y > 0:
            y = 1
        X_tensor = torch.tensor(X, dtype=torch.float32)
        X_tensor = torch.unsqueeze(X_tensor, dim=0)
        y_tensor = torch.tensor(y, dtype=torch.long)
        if self.transform:
            X_tensor = self.transform(X_tensor)
        return X_tensor, y_tensor
    
    def __len__(self):
        return int(self.total_size)