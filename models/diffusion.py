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

        # Usa el mismo device que los datos
        device = masked.device

        # Asegura que todos los inputs están en GPU/CPU correcto
        masked = masked.to(device)
        edges = edges.to(device)
        mask = mask.to(device)

        # Pone el modelo en modo evaluación (sin dropout, etc.)
        model.eval()

        # Tamaño del batch
        n = masked.shape[0]

        # =========================
        # INICIO: ruido puro
        # =========================
        # x_T ~ N(0,1)
        x = torch.randn((n, 3, 256, 256), device=device)

        # =========================
        # LOOP INVERSO (T → 1)
        # =========================
        for t in reversed(range(1, self.T)):

            # Tensor con el timestep actual para todo el batch
            t_tensor = torch.full((n,), t, device=device, dtype=torch.long)

            # =========================
            # INPUT CONDICIONADO
            # =========================
            # x_t (ruido actual) + imagen dañada + bordes + máscara
            input_model = torch.cat([x, masked, edges, mask], dim=1)

            # El modelo predice el ruido presente en x_t
            predicted_noise = model(input_model, t_tensor)

            # =========================
            # PARÁMETROS DEL PASO t
            # =========================
            alpha = self.alpha[t]          # cuánto se conserva
            alpha_hat = self.alpha_hat[t] # acumulado
            beta = self.beta[t]           # ruido del paso

            # =========================
            # COEFICIENTES DDPM
            # =========================
            # Factor de normalización
            coef1 = 1 / torch.sqrt(alpha)

            # Factor que elimina el ruido predicho
            coef2 = (1 - alpha) / torch.sqrt(1 - alpha_hat)

            # =========================
            # RUIDO ESTOCÁSTICO
            # =========================
            # Añade aleatoriedad (necesario para muestreo)
            if t > 1:
                noise = torch.randn_like(x)
            else:
                # En el último paso ya no añadimos ruido
                noise = torch.zeros_like(x)

            # =========================
            # PASO INVERSO
            # =========================
            # x_{t-1} = quitar ruido + añadir pequeña variación
            x = coef1 * (x - coef2 * predicted_noise) + torch.sqrt(beta) * noise

        # Vuelve a modo entrenamiento
        model.train()

        # Limita valores entre 0 y 1 (imagen válida)
        return x.clamp(0, 1)