# -*- coding: utf-8 -*-

"""Trainer for Recurrent CNN.

This script contains the following classes:
    * RecurrentCNNTrainer: Trainer class for Recurrent CNN.
"""

__author__ = "Mir Sazzat Hossain"

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from basicsr.metrics import calculate_psnr, calculate_ssim
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.discriminator import Discriminator
from models.perceptual_loss import VGG16PerceptualLoss
from models.recur_cnn import RecurCNN
from utils.div2k_dataset import DIV2KDataset
from utils.test_dataset import SRTestDataset


class RecurrentCNNTrainer:
    """Trainer for Recurrent CNN."""

    def __init__(
        self,
        data_dir: str,
        train_dataset_name: str,
        test_dataset_names: list,
        scale_factor: int = 4,
        patch_size: int = 48,
        batch_size: int = 16,
        overlap_height_ratio: float = 0.,
        overlap_width_ratio: float = 0.,
        width: int = 64,
        depth: int = 13,
        num_epochs: int = 100,
        lr: float = 1e-4,
        num_workers: int = 4,
        device: str = "cpu",
        log_dir: str = "logs",
        result_dir: str = "results",
        save_interval: int = 10,
        add_discriminator: bool = False,
    ) -> None:
        """
        Init function.

        :param data_dir: data directory.
        :type data_dir: str
        :param train_dataset_name: train dataset name.
        :type train_dataset_name: str
        :param test_dataset_names: test dataset names.
        :type test_dataset_names: list
        :param scale_factor: scale factor.
        :type scale_factor: int
        :param patch_size: patch size.
        :type patch_size: int
        :param batch_size: batch size.
        :type batch_size: int
        :param overlap_height_ratio: overlap height ratio.
        :type overlap_height_ratio: float
        :param overlap_width_ratio: overlap width ratio.
        :type overlap_width_ratio: float
        :param width: model width.
        :type width: int
        :param depth: model depth.
        :type depth: int
        :param num_epochs: number of epochs.
        :type num_epochs: int
        :param lr: learning rate.
        :type lr: float
        :param num_workers: number of workers.
        :type num_workers: int
        :param device: device.
        :type device: str
        :param log_dir: log directory.
        :type log_dir: str
        :param result_dir: result directory.
        :type result_dir: str
        :param save_interval: save interval.
        :type save_interval: int
        :param add_discriminator: add discriminator.
        :type add_discriminator: bool
        """
        self.data_dir = data_dir
        self.train_dataset_name = train_dataset_name
        self.test_dataset_names = test_dataset_names
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.width = width
        self.depth = depth
        self.num_epochs = num_epochs
        self.lr = lr
        self.num_workers = num_workers
        self.device = torch.device(device)
        self.log_dir = log_dir
        self.result_dir = result_dir
        self.save_interval = save_interval
        self.add_discriminator = add_discriminator

        # Create save directory.
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        # Cleate log directory.
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Create model.
        self.model = RecurCNN(
            scale_factor=self.scale_factor,
            width=self.width,
            depth=self.depth,
        ).to(self.device)

        # count number of trainable parameters
        num_params = sum(p.numel()
                         for p in self.model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params}")

        # Add loss functions
        self.criterion_mse = nn.MSELoss().to(self.device)
        self.criterion_perceptual = VGG16PerceptualLoss(
            device=self.device,
        ).to(self.device)

        # Add optimizer.
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
        )

        # Add discriminator.
        if self.add_discriminator:
            self.discriminator = Discriminator().to(self.device)
            self.criterion_bce = nn.BCELoss().to(self.device)
            self.optimizer_discriminator = optim.Adam(
                self.discriminator.parameters(),
                lr=1e-5,
            )

    def initiate_writer(self, train: bool = True) -> SummaryWriter:
        """
        Initiate writer.

        :param train: train flag.
        :type train: bool

        :return: Summary writer.
        :rtype: SummaryWriter
        """
        # Find the latest run.
        self.run_version = 0
        while os.path.exists(
            os.path.join(self.log_dir, f"run_{self.run_version}"),
        ):
            self.run_version += 1

        if train:
            self.log_dir = os.path.join(
                self.log_dir, f"run_{self.run_version}"
            )
            self.result_dir = os.path.join(
                self.result_dir,
                f"run_{self.run_version}",
            )

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        return SummaryWriter(self.log_dir)

    def train(self) -> None:
        """Train function."""
        # Load dataset.
        train_dataloader = self.data_loader(
            dataset_name=self.train_dataset_name,
            mode="train"
        )

        # Initiate writer.
        writer = self.initiate_writer()

        # Train loop.
        best_loss = float("inf")
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            running_loss_mse = 0.0
            running_loss_perceptual = 0.0

            loop = tqdm.tqdm(
                train_dataloader,
                total=len(train_dataloader),
                leave=False
            )
            for i, (lr, hr) in enumerate(loop):
                # Move to device.
                lr = lr.to(self.device)
                hr = hr.to(self.device)

                # Forward.
                sr = self.model(lr)

                # Train discriminator.
                if self.add_discriminator:
                    self.optimizer_discriminator.zero_grad()
                    fake = self.discriminator(sr.detach())
                    real = self.discriminator(hr)
                    loss_discriminator = self.criterion_bce(
                        fake,
                        torch.zeros_like(fake).to(self.device),
                    ) + self.criterion_bce(
                        real,
                        torch.ones_like(real).to(self.device),
                    )
                    loss_discriminator /= 2

                    loss_discriminator.backward()
                    self.optimizer_discriminator.step()

                # Train generator.
                self.optimizer.zero_grad()

                # Compute loss.
                generator_loss = self.criterion_bce(
                    self.discriminator(sr),
                    torch.ones_like(self.discriminator(sr)).to(self.device),
                ) if self.add_discriminator else 0
                loss_mse = self.criterion_mse(sr, hr)
                loss_perceptual = self.criterion_perceptual(sr, hr)
                loss = loss_mse + loss_perceptual * 0.01
                loss += generator_loss * 0.001

                # new code
                running_loss_mse += loss_mse.item()
                running_loss_perceptual += loss_perceptual.item()

                loss.backward()
                self.optimizer.step()

                # Update progress bar.
                loop.set_description(f"Epoch [{epoch + 1}/{self.num_epochs}]")
                loop.set_postfix(loss=loss.item())

                # Update running loss.
                running_loss += loss.item()

            # Compute running loss.
            running_loss /= len(train_dataloader)
            running_loss_mse /= len(train_dataloader)
            running_loss_perceptual /= len(train_dataloader)

            # Log loss.
            writer.add_scalar("Loss/train", running_loss, epoch)
            writer.add_scalar("Loss/MSE", running_loss_mse, epoch)
            writer.add_scalar(
                "Loss/Perceptual", running_loss_perceptual, epoch
            )

            # Save model.
            if (epoch + 1) % self.save_interval == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": running_loss,
                }, os.path.join(self.log_dir, "model.pth"))

                if self.add_discriminator:
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.discriminator.state_dict(),
                        "optimizer_state_dict":
                            self.optimizer_discriminator.state_dict(),
                        "loss": running_loss,
                    }, os.path.join(self.log_dir, "discriminator.pth"))

            # Save best model.
            if running_loss < best_loss:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": running_loss,
                }, os.path.join(self.log_dir, "best_model.pth"))

                if self.add_discriminator:
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.discriminator.state_dict(),
                        "optimizer_state_dict":
                            self.optimizer_discriminator.state_dict(),
                        "loss": running_loss,
                    }, os.path.join(self.log_dir, "best_discriminator.pth"))

                best_loss = running_loss

    def test(
            self,
            model_path: str = None,
            dataset_name: str = None,
            epoch: int = None
    ) -> None:
        """
        Test function.

        :param model_path: path to model.
        :type model_path: str
        :param epoch: epoch of model.
        :type epoch: int
        :type model_path: str
        """
        if model_path is not None:
            self.model_path = model_path
            self.log_dir = os.path.dirname(model_path)
            self.run_version = os.path.basename(self.log_dir).split("_")[-1]
            if "run" not in self.result_dir:
                self.result_dir = os.path.join(
                    self.result_dir,
                    f"run_{self.run_version}",
                )

            writer = SummaryWriter(self.log_dir)
        else:
            writer = self.initiate_writer(train=False)

        # Load model.
        best_model = torch.load(
            os.path.join(self.log_dir, "best_model.pth"),
        )
        self.model.load_state_dict(best_model["model_state_dict"])
        self.model.eval()

        # Load test dataset.
        test_loader = self.data_loader(
            dataset_name=dataset_name,
            mode="test"
        )

        # Test loop.
        with torch.no_grad():
            sum_time = 0
            for k, (lr, image_paths) in enumerate(test_loader):
                # Move to device.
                lr = lr.to(self.device)

                # start_time = time.time()

                # Forward.
                sr = self.model(lr)

                # end_time = time.time()
                # sum_time += end_time - start_time

            # print(f"Total time: {sum_time}")

            # Save images.
            for i in range(sr.shape[0]):
                # convert to numpy.
                sr_img = sr[i].cpu().numpy().transpose(1, 2, 0)

                # Convert to PIL image.
                sr_img = Image.fromarray(
                    (sr_img * 255).astype(np.uint8)
                )

                # Save image.
                image_dir = os.path.join(
                    self.result_dir,
                    dataset_name,
                    'outputs',
                    os.path.dirname(image_paths[i]).split("/")[-1]
                )
                os.makedirs(image_dir, exist_ok=True)
                sr_img.save(
                    os.path.join(
                        self.result_dir,
                        dataset_name,
                        'outputs',
                        os.path.dirname(image_paths[i]).split("/")[-1],
                        os.path.basename(image_paths[i])
                    )
                )

        # Log images.
        paths = os.listdir(
            os.path.join(self.result_dir, dataset_name, 'outputs')
        )
        paths.sort()

        hr_paths = os.listdir(
            os.path.join(
                self.data_dir,
                dataset_name,
                "hr"
            )
        )
        hr_paths.sort()
        start_idx = len(hr_paths) - len(paths) + 1
        hr_paths = hr_paths[start_idx-1:]

        # Calculate PSNR.
        sum_psnr = 0
        sum_ssim = 0
        for i in range(len(paths)):
            sr_img = Image.open(
                os.path.join(
                    self.result_dir,
                    dataset_name,
                    'outputs',
                    paths[i]
                )
            )
            hr_img = Image.open(
                os.path.join(
                    self.data_dir,
                    dataset_name,
                    "hr",
                    hr_paths[i]
                )
            ).convert("RGB")

            psnr = calculate_psnr(
                np.array(sr_img),
                np.array(hr_img),
                crop_border=5
            )

            ssim = calculate_ssim(
                np.array(sr_img),
                np.array(hr_img),
                crop_border=5
            )

            writer.add_scalar(
                f"PSNR/test/{dataset_name}",
                psnr, start_idx + i
            )
            writer.add_scalar(
                f"SSIM/test/{dataset_name}",
                ssim, start_idx + i
            )
            sum_psnr += psnr
            sum_ssim += ssim

        avg_psnr = sum_psnr / len(paths)
        avg_ssim = sum_ssim / len(paths)
        writer.add_scalar(
            f"PSNR/test_avg/{dataset_name}", avg_psnr,
            self.run_version if epoch is None else epoch
        )
        writer.add_scalar(
            f"SSIM/test_avg/{dataset_name}", avg_ssim,
            self.run_version if epoch is None else epoch
        )

    def data_loader(
            self,
            dataset_name: str = None,
            mode: str = "train"
    ) -> DataLoader:
        """
        Load data.

        :param mode: train or test mode.
        :type mode: str

        :return: Train data loader.
        :rtype: DataLoader
        """
        if mode == "train":
            dataset = DIV2KDataset(
                data_dir=self.data_dir,
                dataset_name=dataset_name,
                scale_factor=self.scale_factor,
                patch_size=self.patch_size
            )
        else:
            dataset = SRTestDataset(
                data_dir=self.data_dir,
                dataset_name=dataset_name,
                scale_factor=self.scale_factor,
            )

        # Create data loader.
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True if mode == "train" else False,
            num_workers=self.num_workers,
        )

        return dataloader
