import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch

class MedSegDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx])
        mask = np.load(self.mask_paths[idx])
        if self.transform:
            image, mask = self.transform(image, mask)
        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1), torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

def get_non_iid_splits(data_dir, num_clients=3):
    # Assume data_dir has imgs/ and gts/ with .npy files
    img_dir = os.path.join(data_dir, 'imgs')
    gt_dir = os.path.join(data_dir, 'gts')
    img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.npy')])
    gt_files = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith('.npy')])
    # Simulate non-IID: Hospital 1: first 1/3 (mostly benign), Hospital 2: middle 1/3 (mostly malignant), Hospital 3: last 1/3 (mixed)
    n = len(img_files)
    split1 = img_files[:n//3], gt_files[:n//3]
    split2 = img_files[n//3:2*n//3], gt_files[n//3:2*n//3]
    split3 = img_files[2*n//3:], gt_files[2*n//3:]
    return [split1, split2, split3]

def get_data_loaders(client_id, data_dir, batch_size=16):
    splits = get_non_iid_splits(data_dir)
    img_paths, mask_paths = splits[client_id]
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(img_paths, mask_paths, test_size=0.2, random_state=42)
    train_dataset = MedSegDataset(train_imgs, train_masks)
    val_dataset = MedSegDataset(val_imgs, val_masks)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
