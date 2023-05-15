import os
import pandas as pd
import numpy as np
import cv2
from typing import Any, Tuple
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class MedSamDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_col: str,
        mask_col: str,
        image_dir: Any = None,
        mask_dir: str = None,
        image_size: Tuple = (256, 256),
    ):
        """
        PyTorch dataset class for loading image,mask and bbox pairs from a dataframe.
        The dataframe will need to have atleast two columns for the image and mask file names. The columns can either have the full or relative
        path of the images or just the file names.
        If only file names are given in the columns, the `image_dir` and `mask_dir` arguments should be specified.

        Args:
            df (pd.DataFrame): the pandas dataframe object
            image_col (str): the name of the column on the dataframe that holds the image file names.
            mask_col (str): the name of the column on the dataframe that holds the mask file names.
            image_dir (Any, optional): Path to the input image directory. Defaults to None.
            mask_dir (str, optional): Path to the mask images directory. Defaults to None.
            image_size (Tuple, optional): image size. Defaults to (256, 256).
        """
        self.df = df
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_col = image_col
        self.mask_col = mask_col
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # read dataframe row
        row = self.df.iloc[idx]
        # If the `image_dir` attribute is set, the path will be relative to that directory.
        # Otherwise, the path will be the value of the `row[self.image_col]` attribute.
        image_file = (
            os.path.join(self.image_dir, row[self.image_col])
            if self.image_dir
            else row[self.image_col]
        )
        mask_file = (
            os.path.join(self.mask_dir, row[self.mask_col])
            if self.mask_dir
            else row[self.mask_col]
        )

        if not os.path.exists(image_file):
            raise FileNotFoundError(f"Couldn't find image {image_file}")
        if not os.path.exists(mask_file):
            raise FileNotFoundError(f"Couldn't find image {mask_file}")

        # read image and mask files
        image_data = cv2.imread(image_file)
        # read mask as gray scale
        mask_data = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        return self._preprocess(image_data, mask_data)

    def _preprocess(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Threshold mask to binary
        mask = cv2.threshold(mask, 127.0, 255.0, cv2.THRESH_BINARY)[1]
        # convert to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        # min-max normalize and scale
        image = (image - image.min()) / (image.max() - image.min()) * 255.0
        # resize
        image = TF.resize(image, self.image_size, antialias=True)
        mask = TF.resize(mask, self.image_size, antialias=True)

        bbox = self._get_bbox(mask)

        return image, mask, bbox

    def _get_bbox(self, mask: torch.Tensor) -> torch.Tensor:
        _, y_indices, x_indices = torch.where(mask > 0)

        x_min, y_min = (x_indices.min(), y_indices.min())
        x_max, y_max = (x_indices.max(), y_indices.max())

        # add perturbation to bounding box coordinates
        H, W = mask.shape[1:]
        # add perfurbation to the bbox
        assert H == W, f"{W} and {H} are not equal size!!"
        x_min = max(0, x_min - np.random.randint(0, 10))
        x_max = min(W, x_max + np.random.randint(0, 10))
        y_min = max(0, y_min - np.random.randint(0, 10))
        y_max = min(H, y_max + np.random.randint(0, 10))

        return torch.tensor([x_min, y_min, x_max, y_max])
