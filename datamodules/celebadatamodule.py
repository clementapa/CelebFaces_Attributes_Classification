from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
import os

from datasets.celeba import MyCelebA
from hparams import InferenceParams
from datasets.inference_dataset import CustomImageDataset

class CelebADataModule(LightningDataModule):
    def __init__(self, config, train):
        super().__init__()

        self.root = config.root_dataset
        self.batch_size = config.batch_size
        self.transform = self.get_transforms(config.input_size)
        self.num_workers = config.num_workers

        self.attr_dict = None

    def get_transforms(self, input_size):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return {
            "train": transforms.Compose(
                [
                    transforms.RandomVerticalFlip(p=0.4),
                    transforms.RandomHorizontalFlip(p=0.4),
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
            "val/test/pred": transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
        }

    def setup(self, stage=None):
        # transforms
        # split dataset
        if stage in (None, "fit"):
            self.train = MyCelebA(
                self.root, split="train", transform=self.transform["train"]
            )
            self.val = MyCelebA(
                self.root,
                split="valid",
                transform=self.transform["val/test/pred"],
            )

        if stage == "test":
            self.test = MyCelebA(
                self.root,
                split="test",
                transform=self.transform["val/test/pred"],
            )
        if stage == "predict":
            self.predict = CustomImageDataset(
                self.root,
                transform=self.transform["val/test/pred"],
            )

    def train_dataloader(self):
        train = DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
        return train

    def val_dataloader(self):
        val = DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return val

    def test_dataloader(self):
        test = DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return test

    def predict_dataloader(self):
        predict = DataLoader(
            self.predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return predict
