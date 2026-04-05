import torch

def add_noise(x, t):
    noise = torch.randn_like(x)
    alpha_t = 0.9 ** t
    noisy = (alpha_t**0.5)*x + ((1-alpha_t)**0.5)*noise
    return noisy, noise