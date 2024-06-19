# -*- coding: utf-8 -*-

"""DIV2K dataset for image super-resolution.

DIV2K dataset is used for image super-resolution. It contains 900 training
images, all of which are high resolution (HR) images.We provide the code for
generating patches from the DIV2K dataset and the patches are used for
training the model.

This script contains the following classes:
    * DIV2KDataset: DIV2K dataset class.

This script contains the following functions:
    * calculate_slices: calculate slices for overlapping patches.
"""

__author__ = "Mir Sazzat Hossain"


import glob
import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class DIV2KDataset(Dataset):
    """DIV2K dataset with overlapping patches."""

    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        scale_factor: int = 4,
        patch_size: int = 48
    ) -> None:
        """
        Init function.

        :param data_dir: str, data directory.
        :type data_dir: str
        :param dataset_name: str, dataset name.
        :type dataset_name: str
        :param scale_factor: int, scale factor.
        :type scale_factor: int
        :param patch_size: int, patch size.
        :type patch_size: int
        """
        super(DIV2KDataset, self).__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.scale_factor = scale_factor
        self.patch_size = patch_size

        self.hr_paths = []

        self.transform_hr = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ])

        self.transform_lr = transforms.Compose([
            transforms.Resize(
                self.patch_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True
            ),
        ])

        self._load_images()

    def _load_images(self) -> None:
        """Load images."""
        hr_paths = glob.glob(
            os.path.join(
                self.data_dir,
                self.dataset_name,
                f"hr_{self.scale_factor}x"
            ) + "/**/*.png"
        )

        self.hr_paths = sorted(hr_paths)

    def __len__(self) -> int:
        """Length."""
        return len(self.hr_paths)

    def __getitem__(self, index) -> tuple:
        """Get item."""
        hr_image = Image.open(self.hr_paths[index])

        hr_image = self.transform_hr(hr_image)
        lr_image = self.transform_lr(hr_image)

        return lr_image, hr_image


def calculate_slices(
    image_height: int,
    image_width: int,
    patch_height: int = 48,
    patch_width: int = 48,
    overlap_height_ratio: float = 0.75,
    overlap_width_ratio: float = 0.75,
) -> list[list[int]]:
    """
    Calculate slices for overlapping patches.

    :param image_height: int, image height.
    :type image_height: int
    :param image_width: int, image width.
    :type image_width: int
    :param patch_height: int, patch height.
    :type patch_height: int
    :param patch_width: int, patch width.
    :type patch_width: int
    :param overlap_height_ratio: float, overlap height ratio.
    :type overlap_height_ratio: float
    :param overlap_width_ratio: float, overlap width ratio.
    :type overlap_width_ratio: float

    :return: list[list[int]], slices.
    :rtype: list[list[int]]
    """
    slice_bboxes = []
    y_max = y_min = 0
    y_overlap = int(patch_height * overlap_height_ratio)
    x_overlap = int(patch_width * overlap_width_ratio)
    while y_max < image_height:
        x_max = x_min = 0
        y_max = y_min + patch_height
        if y_max > image_height:
            y_min = image_height - patch_height
            y_max = image_height
        while x_max < image_width:
            x_max = x_min + patch_width
            if x_max > image_width:
                x_min = image_width - patch_width
                x_max = image_width
            slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes
