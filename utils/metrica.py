import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

'''el archivo va proporcionar metricas de calidad de imagen, para ello usaremos PSNR y SSIM
siendo PSNR mide la relacionm de señal maxima y el ruido. SSIM compara la luminancia, contraste
y estrctura de la foto original con la reconstruida.
'''

def calcular_metricas(imagenes_reales, reconoc_img):
    '''
    calcularemos PSNR y SSIM entre las imágenes reales y las reconstruidas.
    '''

    dis = imagenes_reales.device
    rec_img = reconoc_img.to(dis)

    rec_img = torch.clamp(rec_img, 0.0, 1.0)
    imagenes_reales = torch.clamp(imagenes_reales, 0.0, 1.0)

    psnr_metrica = PeakSignalNoiseRatio(data_range=1.0).to(dis)
    ssim_metrica = StructuralSimilarityIndexMeasure(data_range=1.0).to(dis)

    psnr = psnr_metrica(rec_img, imagenes_reales)
    ssim = ssim_metrica(rec_img, imagenes_reales)

    return psnr.item(), ssim.item()