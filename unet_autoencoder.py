import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Resize, RandomHorizontalFlip, RandomRotation
import matplotlib.pyplot as plt

class UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        before_pool = x
        x = self.pool(x)
        return x, before_pool

class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoder, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        if x.shape != skip_connection.shape:
            x = F.pad(x, [0, skip_connection.shape[3] - x.shape[3], 0, skip_connection.shape[2] - x.shape[2]])
        x = torch.cat((x, skip_connection), dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class UNetAutoencoder(pl.LightningModule):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetAutoencoder, self).__init__()
        self.encoder1 = UNetEncoder(in_channels, 64)
        self.encoder2 = UNetEncoder(64, 128)
        self.encoder3 = UNetEncoder(128, 256)
        self.encoder4 = UNetEncoder(256, 512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        self.decoder1 = UNetDecoder(1024, 512)
        self.decoder2 = UNetDecoder(512, 256)
        self.decoder3 = UNetDecoder(256, 128)
        self.decoder4 = UNetDecoder(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x, skip1 = self.encoder1(x)
        x, skip2 = self.encoder2(x)
        x, skip3 = self.encoder3(x)
        x, skip4 = self.encoder4(x)

        x = self.bottleneck(x)

        x = self.decoder1(x, skip4)
        x = self.decoder2(x, skip3)
        x = self.decoder3(x, skip2)
        x = self.decoder4(x, skip1)

        x = self.final_conv(x)
        return x

    def training_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self.forward(x)
        mse_loss = F.mse_loss(y_hat, x)
        self.log('train_mse_loss', mse_loss)

        # Mostrar imágenes de entrada y salida
        if batch_idx == 0:
            self.show_images(x, y_hat, 'train')

        return mse_loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self.forward(x)
        mse_loss = F.mse_loss(y_hat, x)
        self.log('val_mse_loss', mse_loss)

        # Mostrar imágenes de entrada y salida
        if batch_idx == 0:
            self.show_images(x, y_hat, 'val')

        return mse_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]

    def show_images(self, input_imgs, output_imgs, phase):
        input_imgs = input_imgs.cpu().detach().numpy()
        output_imgs = output_imgs.cpu().detach().numpy()

        # Normalizar las imágenes al rango [0, 1]
        input_imgs = (input_imgs - input_imgs.min()) / (input_imgs.max() - input_imgs.min())
        output_imgs = (output_imgs - output_imgs.min()) / (output_imgs.max() - output_imgs.min())

        for i in range(3):  # Mostrar las primeras 3 imágenes
            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(input_imgs[i].transpose((1, 2, 0)))
            axes[0].set_title('Input Image')
            axes[0].axis('off')
            
            axes[1].imshow(output_imgs[i].transpose((1, 2, 0)))
            axes[1].set_title('Output Image')
            axes[1].axis('off')

            plt.show()
            plt.close(fig)

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=32, img_size=128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size

        self.transform = Compose([
            Resize((self.img_size, self.img_size)),
            RandomHorizontalFlip(),
            RandomRotation(10),
            ToTensor()
        ])

    def setup(self, stage=None):
        dataset = ImageFolder(self.data_dir, transform=self.transform)
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

if __name__ == "__main__":
    # Asegúrate de especificar la ruta correcta a tu carpeta de imágenes
    data_dir = r'C:\Users\Tony\OneDrive\Escritorio\I semestre 2024\IA\Notebooks\proyecto3\Data\train'
    data_module = CustomDataModule(data_dir=data_dir, img_size=128)
    model = UNetAutoencoder()

    trainer = pl.Trainer(max_epochs=10, accelerator="gpu", default_root_dir=r'C:\Users\Tony\Documents\lightning_logs')
    trainer.fit(model, data_module)
