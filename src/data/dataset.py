from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms


def _load_rgb_image(image_path: Path) -> Image.Image:
    """Load an RGB image from disk."""
    with Image.open(image_path) as image:
        return image.convert("RGB")


def _is_valid_image(image_path: Path) -> bool:
    """Return whether an image can be identified and decoded by Pillow."""
    try:
        with Image.open(image_path) as image:
            image.verify()
        return True
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return False


class ImageClassificationDataset(Dataset):
    """Dataset that reads image paths and labels from a split CSV file."""

    def __init__(
        self,
        split_csv: Path | str,
        class_names: Sequence[str],
        transform: Optional[Callable] = None,
        validate_images: bool = False,
    ) -> None:
        self.split_csv = Path(split_csv)
        if not self.split_csv.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_csv}")

        self.records = pd.read_csv(self.split_csv)
        required_columns = {"image_path", "label"}
        missing_columns = required_columns - set(self.records.columns)
        if missing_columns:
            raise ValueError(
                f"Split file {self.split_csv} is missing columns: {sorted(missing_columns)}"
            )

        self.invalid_image_paths: list[str] = []
        if validate_images:
            valid_images = self.records["image_path"].map(
                lambda image_path: _is_valid_image(Path(image_path))
            )
            self.invalid_image_paths = (
                self.records.loc[~valid_images, "image_path"].astype(str).tolist()
            )
            self.records = self.records.loc[valid_images].reset_index(drop=True)
            if self.records.empty:
                raise ValueError(f"No valid images found in {self.split_csv}")

        self.class_names = list(class_names)
        self.class_to_idx = {class_name: index for index, class_name in enumerate(self.class_names)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records.iloc[index]
        image_path = Path(record["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = _load_rgb_image(image_path)
        if self.transform is not None:
            image = self.transform(image)

        label_index = self.class_to_idx[str(record["label"])]
        label = torch.tensor(float(label_index), dtype=torch.float32)
        return image, label, str(image_path)


class SuperResolutionPairDataset(Dataset):
    """Dataset that returns low-resolution inputs paired with high-resolution targets."""

    def __init__(
        self,
        split_csv: Path | str,
        class_names: Sequence[str],
        low_resolution_size: int = 32,
        high_resolution_size: int = 128,
        low_resolution_transform: Optional[Callable] = None,
        high_resolution_transform: Optional[Callable] = None,
    ) -> None:
        self.split_csv = Path(split_csv)
        if not self.split_csv.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_csv}")

        self.records = pd.read_csv(self.split_csv)
        required_columns = {"image_path", "label"}
        missing_columns = required_columns - set(self.records.columns)
        if missing_columns:
            raise ValueError(
                f"Split file {self.split_csv} is missing columns: {sorted(missing_columns)}"
            )

        self.class_names = list(class_names)
        self.class_to_idx = {class_name: index for index, class_name in enumerate(self.class_names)}
        self.low_resolution_size = low_resolution_size
        self.high_resolution_size = high_resolution_size
        self.low_resolution_transform = low_resolution_transform
        self.high_resolution_transform = high_resolution_transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        record = self.records.iloc[index]
        image_path = Path(record["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = _load_rgb_image(image_path)
        high_resolution_image = image.resize((self.high_resolution_size, self.high_resolution_size), Image.BICUBIC)
        low_resolution_image = image.resize((self.low_resolution_size, self.low_resolution_size), Image.BICUBIC)

        if self.high_resolution_transform is not None:
            high_resolution = self.high_resolution_transform(high_resolution_image)
        else:
            high_resolution = transforms.ToTensor()(high_resolution_image)

        if self.low_resolution_transform is not None:
            low_resolution = self.low_resolution_transform(low_resolution_image)
        else:
            low_resolution = transforms.ToTensor()(low_resolution_image)

        if not isinstance(high_resolution, torch.Tensor):
            raise TypeError("High-resolution transform must return a torch.Tensor.")

        if not isinstance(low_resolution, torch.Tensor):
            raise TypeError("Low-resolution transform must return a torch.Tensor.")

        label_index = self.class_to_idx[str(record["label"])]
        label = torch.tensor(float(label_index), dtype=torch.float32)
        return low_resolution, high_resolution, label, str(image_path)
