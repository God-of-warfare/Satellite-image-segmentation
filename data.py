import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import glob
from typing import Tuple, Dict, List


class LandCoverDataset(Dataset):
    """
    Dataset class for Land Cover Segmentation
    """

    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 transform: bool = True,
                 img_size: Tuple[int, int] = (256, 256)):
        """
        Args:
            data_dir (str): Base directory for data
            split (str): 'train' or 'val'
            transform (bool): Whether to apply transforms
            img_size (tuple): Size to resize images to
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.transform = transform

        # Color mapping for land cover classes
        self.COLOR_MAPPING = {
            (0, 255, 255): 0,  # Urban Land
            (255, 255, 0): 1,  # Agriculture Land
            (255, 0, 255): 2,  # Rangeland
            (0, 255, 0): 3,  # Forest Land
            (0, 0, 255): 4,  # Water
            (255, 255, 255): 5,  # Barren Land
            (0, 0, 0): 6  # Unknown
        }

        # Get all image paths
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, '*_sat.jpg')))
        self.mask_paths = sorted(glob.glob(os.path.join(data_dir, '*_mask.png')))

        # Define transforms
        if self.transform:
            self.image_transforms = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

            self.mask_transforms = transforms.Compose([
                transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST)
            ])

            self.common_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomInvert(p=0.5)
                ])


    def __len__(self) -> int:
        return len(self.image_paths)

    def convert_mask_to_labels(self, mask: np.ndarray) -> np.ndarray:
        """Convert RGB mask to class labels"""
        label_mask = np.zeros(mask.shape[:2], dtype=np.int64)

        # Convert RGB mask to label mask
        for rgb, label in self.COLOR_MAPPING.items():
            mask_bool = (mask[..., 0] == rgb[0]) & \
                        (mask[..., 1] == rgb[1]) & \
                        (mask[..., 2] == rgb[2])
            label_mask[mask_bool] = label

        return label_mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get image and mask pair"""
        # Load image and mask
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Verify matching pairs
        img_num = os.path.basename(image_path).split('_')[0]
        mask_num = os.path.basename(mask_path).split('_')[0]
        # assert img_num == mask_num, \
        #     f"Mismatch between image {img_num} and mask {mask_num}"

        # Load image and mask
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.image_transforms(image)
            # image = self.common_transforms(image)
            mask = self.mask_transforms(mask)
            # mask = self.common_transforms(mask)


        # Convert mask to numpy for label conversion
        mask_np = np.array(mask)
        label_mask = self.convert_mask_to_labels(mask_np)

        return image, torch.from_numpy(label_mask)

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset"""
        print("Calculating class weights...")
        label_counts = torch.zeros(len(self.COLOR_MAPPING))

        for idx in range(len(self)):
            _, mask = self.__getitem__(idx)
            for i in range(len(self.COLOR_MAPPING)):
                label_counts[i] += (mask == i).sum()

        # Calculate weights: 1 / (log(c + class_samples))
        # Adding 1.02 to ensure no division by zero
        weights = 1.0 / torch.log(label_counts + 1.02)
        return weights


# Example usage:
def create_dataloaders(data_dir: str,
                       batch_size: int = 16,
                       num_workers: int = 4,
                       img_size: Tuple[int, int] = (256, 256)) -> Tuple[torch.utils.data.DataLoader, torch.Tensor]:
    """
    Create dataset and dataloader
    """
    # Create dataset
    dataset = LandCoverDataset(
        data_dir=data_dir,
        transform=True,
        img_size=img_size
    )

    # Calculate class weights for weighted cross entropy
    class_weights = dataset.get_class_weights()

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader, class_weights


# Verification function
def verify_dataset(data_dir: str, img_size: Tuple[int, int] = (256, 256)):
    """
    Verify dataset by displaying a few samples
    """
    import matplotlib.pyplot as plt

    dataset = LandCoverDataset(data_dir, transform=True, img_size=img_size)

    # Display first few samples
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))

    for i in range(3):
        image, mask = dataset[i]

        # Denormalize image
        img_np = image.numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)

        # Plot
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f'Image {i}')
        axes[i, 1].imshow(mask.numpy())
        axes[i, 1].set_title(f'Mask {i}')

    plt.tight_layout()
    plt.show()

    # Print class distribution
    print("\nClass distribution in first image:")
    unique, counts = np.unique(mask.numpy(), return_counts=True)
    for u, c in zip(unique, counts):
        class_name = [k for k, v in dataset.COLOR_MAPPING.items() if v == u][0]
        print(f"Class {u} ({class_name}): {c} pixels")

