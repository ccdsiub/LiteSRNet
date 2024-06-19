"""Test datasets for Super-Resolution.

This script contains most used test datasets for Super-Resolution. Currently
supported datasets are:

    * Set5
    * Set14
    * Urban100
    * BSD100

I am not the owner of any of these datasets. I have downloaded them from
huggingface datasets library. I have just written a script to load them
for my own use.

The script contains the following classes:

    * SRTestDataset: Super-Resolution test dataset class.
"""

__author__ = "Mir Sazzat Hossain"


import glob
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SRTestDataset(Dataset):
    """Super-Resolution test dataset."""

    def __init__(
        self, data_dir: str,
        dataset_name: str,
        scale_factor: int = 4
    ) -> None:
        """Init function.

        :param data_dir: str, data directory.
        :type data_dir: str
        :param dataset_name: str, dataset name.
        :type dataset_name: str
        :param scale_factor: int, scale factor.
        :type scale_factor: int
        """
        super(SRTestDataset, self).__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.scale_factor = scale_factor

        self.lr_paths = []

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load dataset."""
        lr_dir = os.path.join(
            self.data_dir, self.dataset_name,
            f"lr_{self.scale_factor}"
        )

        self.lr_paths = sorted(glob.glob(lr_dir + "/**/*.png"))

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.lr_paths)

    def __getitem__(self, idx: int) -> tuple:
        """Get item from the dataset.

        :param idx: int, index of the item.
        :type idx: int
        :return: tuple, (lr_image, lr_path)
        :rtype: tuple
        """
        lr_image = Image.open(self.lr_paths[idx]).convert("RGB")
        lr_image = self.transform(lr_image)

        return lr_image, self.lr_paths[idx]
