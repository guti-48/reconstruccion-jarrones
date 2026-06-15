import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CeramicDataset(Dataset):
    def __init__(self, root_dir, edges_dir=None, edge_mode="generated", canny_sigma=0.33):
        if edge_mode not in {"generated", "canny"}:
            raise ValueError("edge_mode debe ser 'generated' o 'canny'")

        self.root_dir = root_dir
        self.edge_mode = edge_mode
        self.canny_sigma = canny_sigma
        self.images_dir = os.path.join(root_dir, "images")
        self.masked_dir = os.path.join(root_dir, "masked_images")
        self.masks_dir = os.path.join(root_dir, "masks")

        if edges_dir is None:
            base_data_dir = os.path.dirname(root_dir)
            self.edges_dir = os.path.join(base_data_dir, "results", "edges")
        else:
            self.edges_dir = edges_dir

        for path in (self.images_dir, self.masked_dir, self.masks_dir):
            if not os.path.isdir(path):
                raise FileNotFoundError(f"No existe el directorio requerido: {path}")

        self.files = sorted(
            file for file in os.listdir(self.images_dir)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        if not self.files:
            raise FileNotFoundError(f"No hay imagenes en {self.images_dir}")

    def __len__(self):
        return len(self.files)

    def _load_generated_edge(self, filename):
        base_name = os.path.splitext(filename)[0]
        pattern = os.path.join(self.edges_dir, f"esqueleto_{base_name}.*")
        matches = glob.glob(pattern)

        if not matches:
            raise FileNotFoundError(
                f"[ERROR CRITICO] Falta el borde de {base_name} en {self.edges_dir}. "
                "Ejecuta genera_bordes.py o usa edge_mode='canny' para entrenar M2."
            )

        edge = cv2.imread(matches[0], cv2.IMREAD_GRAYSCALE)
        if edge is None:
            raise FileNotFoundError(f"No se pudo leer el borde generado: {matches[0]}")

        return edge

    def _build_canny_edge(self, img_gray, mask):
        mask_bin = (mask > 127).astype(np.uint8) * 255
        img_inpainted = cv2.inpaint(img_gray, mask_bin, 3, cv2.INPAINT_TELEA)
        blurred = cv2.GaussianBlur(img_inpainted, (3, 3), 0)

        valid_pixels = blurred[img_gray > 15]
        median = np.median(valid_pixels) if valid_pixels.size else 127
        lower = int(max(0, (1.0 - self.canny_sigma) * median))
        upper = int(min(255, (1.0 + self.canny_sigma) * median))

        edges = cv2.Canny(blurred, lower, upper)
        edges[img_gray <= 15] = 0
        return edges

    def __getitem__(self, idx):
        filename = self.files[idx]

        img = cv2.imread(os.path.join(self.images_dir, filename))
        masked = cv2.imread(os.path.join(self.masked_dir, filename))
        mask = cv2.imread(os.path.join(self.masks_dir, filename), cv2.IMREAD_GRAYSCALE)

        if img is None or masked is None or mask is None:
            raise FileNotFoundError(f"No se pudo cargar el trio de datos para {filename}")

        img = cv2.resize(img, (256, 256))
        masked = cv2.resize(masked, (256, 256))
        mask = cv2.resize(mask, (256, 256))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.edge_mode == "generated":
            edges = cv2.resize(self._load_generated_edge(filename), (256, 256))
        else:
            edges = self._build_canny_edge(img_gray, mask)

        img_gray_tensor = torch.tensor(img_gray).unsqueeze(0).float() / 255.0
        mask_tensor = torch.tensor(mask).unsqueeze(0).float() / 255.0
        edges_tensor = torch.tensor(edges).unsqueeze(0).float() / 255.0

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masked_rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(img_rgb).permute(2, 0, 1).float() / 255.0
        masked_tensor = torch.tensor(masked_rgb).permute(2, 0, 1).float() / 255.0

        # Limitamos la mascara al plato para no entrenar sobre el fondo negro.
        plate_silhouette = (img_gray_tensor > 0.1).float()
        mask_tensor = mask_tensor * plate_silhouette

        return masked_tensor, img_tensor, img_gray_tensor, edges_tensor, mask_tensor
