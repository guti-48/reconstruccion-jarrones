import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class CeramicDataset(Dataset):
    def __init__(self, root_dir):
        self.images_dir = os.path.join(root_dir, "images")
        self.masked_dir = os.path.join(root_dir, "masked_images")

        self.files = os.listdir(self.images_dir)

    def adaptive_canny(self, img_gray, sigma=0.33):
        v = np.median(img_gray)

        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        
        edged = cv2.Canny(img_gray, lower, upper)
        
        return edged

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]

        img = cv2.imread(os.path.join(self.images_dir, filename))
        masked = cv2.imread(os.path.join(self.masked_dir, filename))

        img = cv2.resize(img, (256, 256))
        masked = cv2.resize(masked, (256, 256))
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = self.adaptive_canny(img_gray)
        # convertir a tensor añadiendo la dimensión del canal (1, 256, 256)
        edges_tensor = torch.tensor(edges).unsqueeze(0).float() / 255.0
        
        # textura a color
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masked_rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
        
        # permutar dimensiones a formato PyTorch (3, 256, 256)
        img_tensor = torch.tensor(img_rgb).permute(2, 0, 1).float() / 255.0
        masked_tensor = torch.tensor(masked_rgb).permute(2, 0, 1).float() / 255.0

        # retornamos las 3 variables
        return masked_tensor, img_tensor, edges_tensor