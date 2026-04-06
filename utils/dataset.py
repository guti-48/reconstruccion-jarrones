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
        
        #Umbral adaptativo
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
        masked_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(img_gray, masked_gray)
        _, mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
        
        edge = self.adaptive_canny(img_gray)
        
        edge[mask == 255] = 0
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
        img_gray = torch.tensor(img_gray).unsqueeze(0).float() / 255.0
        edge = torch.tensor(edge).unsqueeze(0).float() / 255.0
        mask = torch.tensor(mask).unsqueeze(0).float() / 255.0

        return img, img_gray, edge, mask