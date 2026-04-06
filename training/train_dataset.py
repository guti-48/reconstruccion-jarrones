import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset import *


def test_dataset():
 
    dataset = CeramicDataset(root_dir='imagenes_procesadas') 
    
    print(f"Total de imágenes en el dataset: {len(dataset)}")
    
    img, img_gray, edge, mask = dataset[0]
    img_np = img.permute(1, 2, 0).numpy()
    
    img_gray_np = img_gray.squeeze().numpy()
    edge_np = edge.squeeze().numpy()
    mask_np = mask.squeeze().numpy()
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(img_np)
    axes[0].set_title('Ground Truth (RGB)')
    axes[0].axis('off')
    
    axes[1].imshow(img_gray_np, cmap='gray')
    axes[1].set_title('Escala de Grises')
    axes[1].axis('off')
    
    axes[2].imshow(edge_np, cmap='gray')
    axes[2].set_title('Canny Adaptativo (Bordes)')
    axes[2].axis('off')
    
    axes[3].imshow(mask_np, cmap='gray')
    axes[3].set_title('Máscara Generada')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_dataset()