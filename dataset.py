import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Resize, RandomHorizontalFlip, RandomRotation

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
