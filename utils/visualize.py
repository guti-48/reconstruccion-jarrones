import matplotlib.pyplot as plt
import torch
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def mostrar_resultados(masked, real, edges, mask, model, diffusion, device=Device):

    model.eval()

    masked = masked.to(device)[0:1]
    real = real.to(device)[0:1]
    edges = edges.to(device)[0:1]
    mask = mask.to(device)[0:1]

    with torch.no_grad():
        reconstruida = diffusion.sample(model, masked, edges, mask)

    def to_numpy(x):
        x = x[0].cpu().permute(1,2,0).numpy()
        return x.clip(0,1)

    plt.figure(figsize=(15,4))

    plt.subplot(1,4,1)
    plt.title("Dañada")
    plt.imshow(to_numpy(masked))

    plt.subplot(1,4,2)
    plt.title("Edges (GAN)")
    plt.imshow(edges[0].cpu(), cmap="gray")

    plt.subplot(1,4,3)
    plt.title("Original")
    plt.imshow(to_numpy(real))

    plt.subplot(1,4,4)
    plt.title("Reconstruida")
    plt.imshow(to_numpy(reconstruida))

    plt.tight_layout()
    plt.show()