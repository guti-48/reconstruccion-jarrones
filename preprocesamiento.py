import cv2, os, random
import numpy as np

FOTOS_DIR = 'img'
OUTPUT_FOLDER = 'imagenes_procesadas'
IMG_SIZE = 256

os.makedirs(f'{OUTPUT_FOLDER}/images', exist_ok=True)
os.makedirs(f'{OUTPUT_FOLDER}/masks', exist_ok=True)
os.makedirs(f'{OUTPUT_FOLDER}/masked_images', exist_ok=True)

###############################################
######                                  #######
######                                  #######
######                                  #######
######                                  #######
###############################################
def genera_mascara(img_shape):
    '''Genera una mascara de daño virtual'''
    mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
    h, w = img_shape[0], img_shape[1]

    #añadiremos grietas
    for _ in range(random.randint(2, 6)):
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2, y2 = random.randint(0, w), random.randint(0, h)
        thickness = random.randint(5, 20)
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)

    for _ in range(random.randint(1, 3)):
        cx, cy = random.randint(0, w), random.randint(0, h)
        axes = (random.randint(10, 40), random.randint(10, 40))
        angle = random.randint(0, 180)
        cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)
        
    return mask

print("Iniciando preprocesamiento de imágenes...")
archivos = [f for f in os.listdir(FOTOS_DIR) if f.lower().endswith(('.jpg', '.jpeg'))]

for count, filename in enumerate(archivos):
    img_path = os.path.join(FOTOS_DIR, filename)
    img = cv2.imread(img_path)
    
    if img is None:
        continue
        
    # 1. Redimensionar a 256x256
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # 2. Generar máscara de daño (255 = daño, 0 = intacto)
    mask = genera_mascara((IMG_SIZE, IMG_SIZE))
    
    # 3. Crear imagen con el daño aplicado (para visualizar, las zonas dañadas en blanco)
    masked_image = img_resized.copy()
    masked_image[mask == 255] = [255, 255, 255] 
    
    # 4. Guardar archivos
    cv2.imwrite(f'{OUTPUT_FOLDER}/images/{filename}', img_resized)
    cv2.imwrite(f'{OUTPUT_FOLDER}/masks/{filename}', mask)
    cv2.imwrite(f'{OUTPUT_FOLDER}/masked_images/{filename}', masked_image)

print(f"Preprocesamiento completado. {len(archivos)} imágenes procesadas.")