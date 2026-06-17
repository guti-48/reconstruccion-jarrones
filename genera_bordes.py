import os
import torch
import cv2
import numpy as np
from utils.models import EdgeModel
from training.train_gan import Config

def clean_and_get_edges(img_gray, mask_bin, sigma=0.33):
    # Crear máscara estricta del fondo para matar el ruido exterior
    _, bg_mask = cv2.threshold(img_gray, 15, 255, cv2.THRESH_BINARY)
    kernel_bg = np.ones((3,3), np.uint8)
    bg_mask = cv2.dilate(bg_mask, kernel_bg, iterations=1) 
    
    # Inpainting para ocultar el rayajo blanco a los ojos de Canny
    img_inpainted = cv2.inpaint(img_gray, mask_bin, 3, cv2.INPAINT_TELEA)
    
    # Desenfocar
    blurred = cv2.GaussianBlur(img_inpainted, (3, 3), 0)
    
    # Mediana solo en la cerámica válida
    pixeles_validos = blurred[bg_mask == 255]
    v = np.median(pixeles_validos) if len(pixeles_validos) > 0 else 127
    
    # Canny natural
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(blurred, lower, upper)
    
    # Limpieza pre-IA
    edges[bg_mask == 0] = 0     # Borrar cualquier borde detectado en el fondo
    edges[mask_bin == 255] = 0  # Vaciar el agujero: la IA debe predecir esto, si le dejamos basura, se confunde
    
    # El hachazo de la regla
    alto, ancho = edges.shape
    edges[int(alto * 0.70):alto, :] = 0
    
    return edges, bg_mask

def remove_small_noise(img, min_size=15):
    # Algoritmo que busca formas y borra las que sean más pequeñas que 'min_size'
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    clean_img = np.zeros_like(img)
    
    # Empezamos en 1 para ignorar el fondo negro (0)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            clean_img[labels == i] = 255
            
    return clean_img

def generar_esqueletos():
    config = Config()
    
    carpeta_imagenes = 'data/processed/masked_images'
    carpeta_mascaras = 'data/processed/masks'
    carpeta_salida = 'data/results/edges'
    ruta_modelo_pth = os.path.join(config.PATH, config.GEN_WEIGHTS)
    
    os.makedirs(carpeta_salida, exist_ok=True)
    
    if not os.path.exists(ruta_modelo_pth):
        print(f"Error: No se encuentra el archivo {ruta_modelo_pth}.")
        return
        
    modelo = EdgeModel(config).to(config.DEVICE)
    checkpoint = torch.load(ruta_modelo_pth, map_location=config.DEVICE)
    pesos_gen = checkpoint['generator'] if 'generator' in checkpoint else checkpoint
    modelo.generator.load_state_dict(pesos_gen)
    modelo.eval()

    archivos = os.listdir(carpeta_imagenes)
    
    with torch.no_grad(): 
        for archivo in archivos:
            ruta_img = os.path.join(carpeta_imagenes, archivo)
            ruta_mask = os.path.join(carpeta_mascaras, archivo)
            
            if not os.path.exists(ruta_mask):
                continue
                
            img = cv2.imread(ruta_img)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = cv2.imread(ruta_mask, cv2.IMREAD_GRAYSCALE)
            
            img_gray = cv2.resize(img_gray, (256, 256))
            mask = cv2.resize(mask, (256, 256))
            
            _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            # PRE-PROCESAMIENTO BLINDADO 
            edges, bg_mask = clean_and_get_edges(img_gray, mask_bin)
            
            # Tensors[cite: 2]
            img_gray_tensor = torch.tensor(img_gray).unsqueeze(0).unsqueeze(0).float().to(config.DEVICE) / 255.0
            edges_tensor = torch.tensor(edges).unsqueeze(0).unsqueeze(0).float().to(config.DEVICE) / 255.0
            mask_tensor = torch.tensor(mask_bin).unsqueeze(0).unsqueeze(0).float().to(config.DEVICE) / 255.0

            # El modelo reemplaza internamente la máscara por 1.0 (blanco puro) para predecir[cite: 1]
            outputs = modelo(img_gray_tensor, edges_tensor, mask_tensor)
            
            # Fusionamos los bordes[cite: 3]
            outputs_merged = (outputs * mask_tensor) + (edges_tensor * (1 - mask_tensor))
            borde_final = outputs_merged.squeeze().cpu().numpy()
            
            # Aumentamos el umbral al 65% para matar las dudas de la IA
            borde_final = (borde_final > 0.65).astype(np.uint8) * 255
            
            # Imponemos el fondo negro y la regla negra
            borde_final[bg_mask == 0] = 0 
            alto, ancho = borde_final.shape
            borde_final[int(alto * 0.70):alto, :] = 0 
            
            # Pasamos la aspiradora para eliminar líneas menores a 15 píxeles de longitud
            borde_final = remove_small_noise(borde_final, min_size=15)
            
            ruta_guardado = os.path.join(carpeta_salida, f"esqueleto_{archivo}")
            cv2.imwrite(ruta_guardado, borde_final)
            print(f" -> Guardado pulido: {archivo}")

if __name__ == "__main__":
    generar_esqueletos()