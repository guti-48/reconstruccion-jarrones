import os, torch

class CheckpointManager:
    '''
    con este archivo lo que biiuscamos es evitar que se sobreescriba modelo bueno sobre malo si la red se satura 
    '''
    def __init__(self, save_dir='checkpoints', model_name='unet_diffusion'):
        self.save_dir = save_dir
        self.model_name = model_name
        self.best_ssim = -1.0 

        os.makedirs(save_dir, exist_ok=True)

    def saveBest(self, model, current_ssim, epoch):
        if current_ssim > self.best_ssim:
            print(f"\nNuevo mejor SSIM --> anterior: {self.best_ssim:.4f} -> nuevo: {current_ssim:.4f}")
            self.best_ssim = current_ssim
                    
            for file in os.listdir(self.save_dir):
                if file.startswith(self.model_name):
                    os.remove(os.path.join(self.save_dir, file))

            save_path = os.path.join(self.save_dir, f"{self.model_name}_epoch{epoch}_ssim{current_ssim:.4f}.pth")
            torch.save(model.state_dict(), save_path)
        else:
            print(f"El SSIM no ha mejorado({current_ssim:.4f}). Mejor SSIM: {self.best_ssim:.4f}")