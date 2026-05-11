import torch

# Selecciona GPU si está disponible, si no CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Diffusion:
    def __init__(self, T=700, beta_start=1e-4, beta_end=0.02, device=device):
        
        # Número total de pasos de difusión (longitud del proceso)
        self.T = T
        
        # Dispositivo donde se ejecuta (CPU o GPU)
        self.device = device

        # =========================
        # NOISE SCHEDULE (beta_t)
        # =========================
        # Secuencia lineal de ruido desde beta_start hasta beta_end
        # Define cuánto ruido se añade en cada paso t
        self.beta = torch.linspace(beta_start, beta_end, T).to(device)

        # =========================
        # ALPHA (cantidad de señal)
        # =========================
        # alpha = 1 - beta → cuánto de la imagen original se conserva en cada paso
        self.alpha = 1. - self.beta

        # =========================
        # ALPHA HAT (acumulado)
        # =========================
        # Producto acumulado de alphas → cuánta señal queda tras varios pasos
        # Permite calcular directamente x_t desde x_0
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    # =========================
    # FORWARD DIFFUSION
    # q(x_t | x_0)
    # =========================
    def add_noise(self, x0, t):
        
        # Genera ruido gaussiano con misma forma que la imagen
        noise = torch.randn_like(x0)

        # Obtiene alpha_hat correspondiente a cada t del batch
        # (cada imagen puede tener un t distinto)
        alpha_hat_t = self.alpha_hat.gather(0, t)

        # √alpha_hat → peso de la imagen original
        sqrt_alpha_hat = torch.sqrt(alpha_hat_t).view(-1,1,1,1)

        # √(1 - alpha_hat) → peso del ruido
        sqrt_one_minus = torch.sqrt(1 - alpha_hat_t).view(-1,1,1,1)

        # =========================
        # FORMULA DDPM
        # x_t = sqrt(alpha_hat)*x0 + sqrt(1-alpha_hat)*noise
        # =========================
        # Genera imagen ruidosa en el paso t
        return sqrt_alpha_hat * x0 + sqrt_one_minus * noise, noise


    # =========================
    # REVERSE DIFFUSION
    # p(x_{t-1} | x_t)
    # =========================
    def sample(self, model, masked, edges, mask):
        device = masked.device

        # Aseguramos tensores en el mismo dispositivo
        masked = masked.to(device)
        edges = edges.to(device)
        mask = mask.to(device) # Máscara: 1 es el agujero (daño), 0 es intacto

        model.eval()
        n = masked.shape[0]

        # INICIO: ruido puro
        x = torch.randn((n, 3, 256, 256), device=device)

        # LOOP INVERSO (T → 1)
        for t in reversed(range(1, self.T)):
            t_tensor = torch.full((n,), t, device=device, dtype=torch.long)

            input_model = torch.cat([x, masked, edges, mask], dim=1)
            predicted_noise = model(input_model, t_tensor)

            alpha = self.alpha[t]          
            alpha_hat = self.alpha_hat[t] 
            beta = self.beta[t]           

            coef1 = 1 / torch.sqrt(alpha)
            coef2 = (1 - alpha) / torch.sqrt(1 - alpha_hat)

            if t > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            # 1. Paso inverso normal (adivina toda la imagen)
            x_pred = coef1 * (x - coef2 * predicted_noise) + torch.sqrt(beta) * noise

            # ==========================================
            # 2. INTERVENCIÓN REPAINT (INPAINTING FORZADO)
            # ==========================================
            # Calculamos cómo se vería la imagen real (masked) en el paso t-1
            if t > 1:
                t_minus_1 = torch.full((n,), t - 1, device=device, dtype=torch.long)
                # Usamos nuestra propia función add_noise para generar la "realidad ruidosa"
                known_noised, _ = self.add_noise(masked, t_minus_1)
            else:
                # En el último paso (t=1), la realidad no tiene ruido
                known_noised = masked

            # 3. Fusión Híbrida:
            # (x_pred * mask) -> Nos quedamos con la imaginación de la red SOLO dentro del agujero
            # (known_noised * (1 - mask)) -> Forzamos la realidad perfecta en el resto del plato
            x = (x_pred * mask) + (known_noised * (1 - mask))

        model.train()
        return x.clamp(0, 1)