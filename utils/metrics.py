import torch
import torch.nn as nn
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

'''
El archivo va proporcionar métricas de calidad de imagen final, para ello usaremos PSNR y SSIM.
PSNR mide la relación de señal máxima y el ruido. 
SSIM compara la luminancia, contraste y estructura de la foto original con la reconstruida.
'''

def calcular_metricas(imagenes_reales, reconoc_img):
    '''
    Calcularemos PSNR y SSIM entre las imágenes reales y las reconstruidas por Difusión.
    '''
    dis = imagenes_reales.device
    rec_img = reconoc_img.to(dis)

    # Aseguramos que los valores estén entre 0 y 1
    rec_img = torch.clamp(rec_img, 0.0, 1.0)
    imagenes_reales = torch.clamp(imagenes_reales, 0.0, 1.0)

    # Instanciamos las métricas de la librería oficial
    psnr_metrica = PeakSignalNoiseRatio(data_range=1.0).to(dis)
    ssim_metrica = StructuralSimilarityIndexMeasure(data_range=1.0).to(dis)

    # Calculamos
    psnr = psnr_metrica(rec_img, imagenes_reales)
    ssim = ssim_metrica(rec_img, imagenes_reales)

    return psnr.item(), ssim.item()


class EdgeAccuracy(nn.Module):
    """
    Mide la Precisión (Precision) y Exhaustividad (Recall) del mapa de bordes generado por la GAN.
    Compara las líneas que ha dibujado el Generador frente a las líneas reales extraídas por Canny.
    """
    def __init__(self, threshold=0.5):
        super(EdgeAccuracy, self).__init__()
        self.threshold = threshold

    def __call__(self, inputs, outputs):
        # Convertimos las predicciones a blanco y negro puro (binario)
        labels = (inputs > self.threshold)
        outputs = (outputs > self.threshold)

        relevant = torch.sum(labels.float())
        selected = torch.sum(outputs.float())

        # Si no había bordes que predecir y no predijo ninguno, es un 100% de acierto
        if relevant == 0 and selected == 0:
            return torch.tensor(1.0), torch.tensor(1.0)

        # Calculamos los verdaderos positivos (líneas que acertó exactamente donde iban)
        true_positive = ((outputs == labels) * labels).float()
        
        # Matemáticas de Recall y Precision
        recall = torch.sum(true_positive) / (relevant + 1e-8)
        precision = torch.sum(true_positive) / (selected + 1e-8)

        return precision, recall