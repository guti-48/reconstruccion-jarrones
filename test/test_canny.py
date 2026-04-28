import cv2
import numpy as np

ruta_imagen = 'data/processed/images/060-IMG_4556.jpg' 

img = cv2.imread(ruta_imagen)
if img is None:
    print(f"¡Error! No encuentro la imagen en {ruta_imagen}")
    exit()

img = cv2.resize(img, (256, 256))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def aplicar_canny_sin_regla(img_gris, kernel_size):
    
    alto, ancho = img_gris.shape
    img_gris[int(alto * 0.65):alto, :] = 0
    
    # Desenfoque gaussiano
    blurred = cv2.GaussianBlur(img_gris, (kernel_size, kernel_size), 0)
    
    # Mediana y umbrales
    pixeles_validos = blurred[blurred > 0]
    v = np.median(pixeles_validos) if len(pixeles_validos) > 0 else 127
    
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    # Canny
    return cv2.Canny(blurred, lower, upper)

# Probamos con el filtro 3x3 que elegimos como ganador
bordes_limpios = aplicar_canny_sin_regla(img_gray.copy(), 3) 

cv2.imwrite('prueba_canny_sin_regla.jpg', bordes_limpios)
print("Imagen guardada como: 'prueba_canny_sin_regla.jpg'")