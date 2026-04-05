import matplotlib.pyplot as plt
import torch

def mostrar_resultados(masked, real, model):
    model.eval()

    with torch.no_grad():
        t = torch.randint(1, 10, (1,))
        
        # Añadir ruido a la imagen real
        noise = torch.randn_like(real)
        alpha_t = 0.9 ** t
        noisy = (alpha_t**0.5)*real + ((1-alpha_t)**0.5)*noise

        # Predicción del modelo
        pred_noise = model(noisy)

        # Aproximación reconstruida
        reconstruida = noisy - pred_noise

    # Convertir a numpy
    def to_numpy(img):
        img = img.squeeze().permute(1,2,0).cpu().numpy()
        return img

    masked_np = to_numpy(masked)
    real_np = to_numpy(real)
    recon_np = to_numpy(reconstruida)

    # Mostrar
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.title("Imagen dañada")
    plt.imshow(masked_np)
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Original")
    plt.imshow(real_np)
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Reconstrucción")
    plt.imshow(recon_np)
    plt.axis("off")

    plt.show()