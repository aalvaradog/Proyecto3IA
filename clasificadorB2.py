import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from unet_autoencoder import *
import wandb
import time

## cambiar dependiendo de la cantidad de datos
autoencoder_checkpoint_path70 = r'C:\Users\tian_\Desktop\lightning_logs\unet_autoencoderB2-70.pth'
autoencoder_checkpoint_path90 = r'C:\Users\tian_\Desktop\lightning_logs\unet_autoencoderB2-90.pth'


# Definir transformaciones para el dataset ajustando el tamaño a 128x128 píxeles
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(128),  # Recorta aleatoriamente la imagen y luego la redimensiona a 128x128 píxeles
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(140),             # Redimensiona la imagen para que el lado más corto tenga 140 píxeles
        transforms.CenterCrop(128),         # Luego recorta la imagen al centro a un tamaño de 128x128 píxeles
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(140),             # Redimensiona la imagen para que el lado más corto tenga 140 píxeles
        transforms.CenterCrop(128),         # Luego recorta la imagen al centro a un tamaño de 128x128 píxeles
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Definir una ResNet basada en el autoencoder
class AutoencoderVGG16B2(pl.LightningModule):
    def __init__(self, data_dir, num_classes):
        super(AutoencoderVGG16B2, self).__init__()
        self.data_dir = data_dir
        self.num_classes = num_classes
        
        
        # Cargar autoencoder
        self.autoencoder = UNetAutoencoder()
        self.autoencoder.load_state_dict(torch.load(autoencoder_checkpoint_path70))
        
       
        
        # Clasificador VGG16
        self.model = models.vgg16(pretrained=True)
        for param in self.model.features.parameters():
            param.requires_grad = False
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        
        self.criterion = nn.CrossEntropyLoss()
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.start_time = time.time()
    def forward(self, x):
       
        x = self.autoencoder(x)
        
        return self.model(x)

    def prepare_data(self):
        # Definir los datasets y dataloaders
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x),
                                                  data_transforms[x])
                          for x in ['train', 'valid', 'test']}
        self.dataloaders = {x: DataLoader(image_datasets[x], batch_size=32,
                                          shuffle=True, num_workers=4)
                            for x in ['train', 'valid', 'test']}
        self.dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
        self.class_names = image_datasets['train'].classes

    def train_dataloader(self):
        return self.dataloaders['train']

    def val_dataloader(self):
        return self.dataloaders['valid']

    def test_dataloader(self):
        return self.dataloaders['test']

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == labels.data).double() / len(labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == labels.data).double() / len(labels)
        self.log('valid_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('valid_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_outputs.append({'valid_loss': loss, 'valid_acc': acc})
        return {'valid_loss': loss, 'valid_acc': acc}
    
         

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['valid_loss'] for x in self.validation_step_outputs]).mean()
        avg_acc = torch.stack([x['valid_acc'] for x in self.validation_step_outputs]).mean()
        self.log('valid_loss', avg_loss, prog_bar=True, logger=True)
        self.log('valid_acc', avg_acc, prog_bar=True, logger=True)
        self.validation_step_outputs.clear()
          # Log to W&B
        wandb.log({
            "Epoch": self.current_epoch,
            "Valid Loss": avg_loss,
            "Valid Acc": avg_acc,
            "Elapsed time": time.time() - self.start_time
        },commit=True)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == labels.data).double() / len(labels)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.test_step_outputs.append({'test_loss': loss, 'test_acc': acc, 'preds': preds, 'labels': labels})
        return {'test_loss': loss, 'test_acc': acc, 'preds': preds, 'labels': labels}

    def on_test_epoch_end(self):
        avg_loss = torch.stack([x['test_loss'] for x in self.test_step_outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in self.test_step_outputs]).mean()
        self.log('test_loss', avg_loss, prog_bar=True, logger=True)
        self.log('test_acc', avg_acc, prog_bar=True, logger=True)
        self.validation_step_outputs.clear()

         # Collect preds and true labels for confusion matrix
        preds = torch.cat([x['preds'] for x in self.test_step_outputs]).cpu()
        trues = torch.cat([x['labels'] for x in self.test_step_outputs]).cpu()

        # Log confusion matrix to W&B
        wandb.log({
            "test_confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=trues.numpy(),
                preds=preds.numpy(),
                class_names=self.class_names,
                title="Confusion Matrix"
            )
        })
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.classifier.parameters(), lr=0.001, momentum=0.9)
        return optimizer