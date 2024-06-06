from unet import UNetAutoencoder
from dataset import CustomDataModule
import pytorch_lightning as pl

if __name__ == "__main__":
    # Asegúrate de especificar la ruta correcta a tu carpeta de imágenes
    data_dir = r'C:\Users\Tony\OneDrive\Escritorio\I semestre 2024\IA\Notebooks\proyecto3\Data\train'
    data_module = CustomDataModule(data_dir=data_dir, img_size=128)
    model = UNetAutoencoder()

    trainer = pl.Trainer(max_epochs=10, accelerator="gpu", default_root_dir=r'C:\Users\Tony\Documents\lightning_logs')
    trainer.fit(model, data_module)