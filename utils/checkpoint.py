import os, torch

class CheckpointManager:
    '''
    con este archivo lo que biiuscamos es evitar que se sobreescriba modelo bueno sobre malo si la red se satura 
    '''
    def __init__(self, save_dir='checkpoints', model_name='modelo_m3_diffusion'):
        self.save_dir = save_dir
        self.model_name = model_name
        # Ahora inicializamos en Infinito, porque buscamos bajar este número a 0
        self.best_metric = float('inf') 

        os.makedirs(save_dir, exist_ok=True)

    def saveBest(self, model, current_loss, epoch):
        # Si el error en el agujero es MENOR que nuestro récord, guardamos
        if current_loss < self.best_metric:
            print(f"\n Nuevo récord en el agujero --> anterior: {self.best_metric:.4f} -> nuevo error: {current_loss:.4f}")
            self.best_metric = current_loss
                    
            for file in os.listdir(self.save_dir):
                if file.startswith(self.model_name):
                    os.remove(os.path.join(self.save_dir, file))

            save_path = os.path.join(self.save_dir, f"{self.model_name}_epoch{epoch}_loss{current_loss:.4f}.pth")
            torch.save(model.state_dict(), save_path)
        else:
            print(f"La calidad dentro del agujero no mejoró ({current_loss:.4f}). Mejor error: {self.best_metric:.4f}")