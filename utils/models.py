import os
import torch
import torch.nn as nn
import torch.optim as optim
from models.networks import EdgeGenerator, Discriminator
from .loss import AdversarialLoss

class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()
        self.name = name
        self.config = config
        self.iteration = 0
        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Cargando pesos del generador %s...' % self.name)
            data = torch.load(self.gen_weights_path, map_location=self.config.DEVICE)
            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        if os.path.exists(self.dis_weights_path):
            print('Cargando pesos del discriminador %s...' % self.name)
            data = torch.load(self.dis_weights_path, map_location=self.config.DEVICE)
            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nGuardando %s...\n' % self.name)
        torch.save({'iteration': self.iteration, 'generator': self.generator.state_dict()}, self.gen_weights_path)
        torch.save({'discriminator': self.discriminator.state_dict()}, self.dis_weights_path)


class EdgeModel(BaseModel):
    def __init__(self, config):
        super(EdgeModel, self).__init__('EdgeModel', config)

        # Generador recibe: Gris (1 canal) + Bordes (1 canal) + Máscara (1 canal) = 3 canales
        # Discriminador recibe: Gris (1 canal) + Bordes (1 canal) = 2 canales
        generator = EdgeGenerator(use_spectral_norm=True)
        discriminator = Discriminator(in_channels=2, use_spectral_norm=True, use_sigmoid=config.GAN_LOSS != 'hinge')
        
        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)
        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(params=generator.parameters(), lr=float(config.LR), betas=(config.BETA1, config.BETA2))
        self.dis_optimizer = optim.Adam(params=discriminator.parameters(), lr=float(config.LR) * float(config.D2G_LR), betas=(config.BETA1, config.BETA2))

    def process(self, images_gray, edges, masks):
        self.iteration += 1

        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # Genera bordes falsos
        outputs = self(images_gray, edges, masks)
        gen_loss = 0
        dis_loss = 0

        # Castigo al Discriminador: Intentar diferenciar reales de falsos
        dis_input_real = torch.cat((images_gray, edges), dim=1)
        dis_input_fake = torch.cat((images_gray, outputs.detach()), dim=1)
        dis_real, dis_real_feat = self.discriminator(dis_input_real)
        dis_fake, dis_fake_feat = self.discriminator(dis_input_fake)
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # Castiga al Generador: Intentar engañar al discriminador
        gen_input_fake = torch.cat((images_gray, outputs), dim=1)
        gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False)
        gen_loss += gen_gan_loss

        # Feature matching loss: para que los bordes sigan la estructura real
        gen_fm_loss = 0
        for i in range(len(dis_real_feat)):
            gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        gen_fm_loss = gen_fm_loss * self.config.FM_LOSS_WEIGHT
        gen_loss += gen_fm_loss

        logs = [("l_d1", dis_loss.item()), ("l_g1", gen_gan_loss.item()), ("l_fm", gen_fm_loss.item())]
        return outputs, gen_loss, dis_loss, logs

    def forward(self, images_gray, edges, masks):
        edges_masked = (edges * (1 - masks))
        images_masked = (images_gray * (1 - masks)) + masks
        inputs = torch.cat((images_masked, edges_masked, masks), dim=1)
        outputs = self.generator(inputs)
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        # Aprende el Falsificador PRIMERO (antes de que el policía cambie sus reglas)
        if gen_loss is not None:
            gen_loss.backward()
        self.gen_optimizer.step()

        # Borramos cualquier "idea" residual en el cerebro del policía para que no se mezcle el entrenamiento
        self.dis_optimizer.zero_grad()

        # Aprende el Policía SEGUNDO
        if dis_loss is not None:
            dis_loss.backward()
        self.dis_optimizer.step()