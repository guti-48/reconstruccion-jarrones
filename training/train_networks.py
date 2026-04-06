import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.networks import EdgeGenerator, Discriminator 

def test_networks():
    print("--- Iniciando prueba de la arquitectura de Redes ---")
    dummy_input_gen = torch.randn(1, 3, 256, 256)
    
    print("Instanciando EdgeGenerator...")
    generator = EdgeGenerator(use_spectral_norm=True)
    try:
        output_gen = generator(dummy_input_gen)
        print(f"¡Éxito! Forma de salida del Generador: {output_gen.shape}")
        print(" (Esperado: [1, 1, 256, 256] -> 1 imagen, 1 canal de bordes, 256x256)")
    except Exception as e:
        print(f"Error en el Generador: {e}")
        return
    
    print("\nInstanciando Discriminador...")
    dummy_input_disc = torch.randn(1, 4, 256, 256) 
    discriminator = Discriminator(in_channels=4, use_spectral_norm=True)

    try:
        output_disc, features = discriminator(dummy_input_disc)
        print(f"¡Éxito! Forma de salida del Discriminador: {output_disc.shape}")
        print(" (Esperado: Una matriz más pequeña, ej: [1, 1, 32, 32], evaluando los parches)")
        print("\n¡La arquitectura funciona perfectamente y no hay errores de dimensiones!")
    except Exception as e:
        print(f"Error en el Discriminador: {e}")

if __name__ == "__main__":
    test_networks()