import torch, os
from torch.utils.data import DataLoader

#Importaciones necesarias de m1 (guti)
from utils.dataset import CeramicDataset
from utils.metrics import calcular_metricas
from utils.checkpoint import CheckpointManager

#Importaciones necesarias de m2(rafa)
from models.networks import EdgeGenerator

#Importaciones necesarias de m3(pusasama)
from models.unet import UNet

#Configuracion inicial
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#cargo mis datos limipios
dataset = CeramicDataset(root_dir='data/processed')
cargaDatos = DataLoader(dataset, batch_size=4, shuffle=True)

#guardia de las metricas
guardian = CheckpointManager(save_dir='checkpoints', model_name='modeloConjunto')

#Inicializacion de redes
'''
TODO 
RAFA Y OUSSAMA AQUI TENEIS QUE INICIALIZAR VUESTRAS REDES
'''

#Inicializacion prinipcal de todo (bucle)