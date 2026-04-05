import os
import cv2
import torch
from torch.utils.data import Dataset

class CeramicDataset(Dataset):
    def __init__(self, root_dir):
        self.images_dir = os.path.join(root_dir, "images")
        self.masked_dir = os.path.join(root_dir, "masked_images")

        self.files = os.listdir(self.images_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]

        img = cv2.imread(os.path.join(self.images_dir, filename))
        masked = cv2.imread(os.path.join(self.masked_dir, filename))

        img = cv2.resize(img, (256, 256))
        masked = cv2.resize(masked, (256, 256))

        img = torch.tensor(img).permute(2,0,1).float() / 255
        masked = torch.tensor(masked).permute(2,0,1).float() / 255

        return masked, img