import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms

# Imports for type hints
from typing import Callable, Optional, Tuple
from torch import Tensor


class FashionMNISTDataset(Dataset):
    """
    Class to handle the dataset to be used for the training of the classification model.
    This class includes the capabilities to download and prepare the data.
    """

    def __init__(
        self,
        train: bool = True,
        dataset_dir: str = "./dataset",
        transform: transforms = None,
    ):
        """
        Initialize the FashionMNISTDataset class.

        Args:
            train (bool): If true, returns training dataset. Returns test dataset otherwise.
            annotations_file (str): Name of the CSV file with the labels.
            dataset_dir (str): Directory where the dataset will be stored.
            transform (transforms): Transforms list to be applied to images when read.
        """

        # Initialize arguments
        self.train = train
        self.split = "train" if train else "test"
        self.dataset_dir = dataset_dir
        self.transform = transform

        # Check if the dataset already exists. If not, download it
        self.dataset_csv = os.path.join(self.dataset_dir, "labels.csv")
        if not os.path.exists(self.dataset_csv):
            print("Downloading dataset")
            self.download_dataset()

        # Read CSV file with annotations
        self.dataset_df = pd.read_csv(self.dataset_csv)
        self.dataset_df = self.dataset_df[self.dataset_df["split"] == self.split]

        # Save dataset length for easy access
        self.data_len = len(self.dataset_df)

        # Save image paths and labels list, to ease reading
        self.image_paths = np.asarray(self.dataset_df.iloc[:, 0])
        self.labels = np.asarray(self.dataset_df.iloc[:, 1])

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return self.data_len

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        """
        Retrieve an image and its label by index.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            Tuple[tensor, int]:
                - image: Transformed image.
                - img_label: Image label.
        """
        # Get image path and label
        img_path = self.image_paths[index]
        img_label = self.labels[index]

        # Open image
        image = Image.open(img_path)

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, img_label

    def download_dataset(self):
        """
        Download the Fashion-MNIST dataset using torchvision library.
        """

        # Download the dataset for training and testing
        train_dataset = datasets.FashionMNIST(
            root=self.dataset_dir, train=True, download=True
        )

        test_dataset = datasets.FashionMNIST(
            root=self.dataset_dir, train=False, download=False
        )

        print("Dataset downloaded!")

        # To save the processed images and annotations files, concatenate the whole dataset
        images = np.concatenate(
            (train_dataset.data.numpy(), test_dataset.data.numpy()), axis=0
        )

        labels = np.concatenate(
            (train_dataset.targets.numpy(), test_dataset.targets.numpy()), axis=0
        )

        # Save split for each image
        splits = ["train"] * len(train_dataset) + ["test"] * len(test_dataset)

        # Create directory to save png images
        image_dir = os.path.join(self.dataset_dir, "images")
        os.makedirs(image_dir, exist_ok=True)

        # Loop through each image, load it and save it as png
        dataset_info = []
        for idx in range(len(images)):
            image = Image.fromarray(images[idx])
            image_path = os.path.join(
                image_dir, f"image_{str(idx)}_label_{labels[idx]}.png"
            )
            image.save(image_path)
            dataset_info.append([image_path, labels[idx], splits[idx]])

        # Create dataframe with images metadata (path, label and split)
        df = pd.DataFrame(dataset_info, columns=["image_path", "label", "split"])

        # Save metadata to CSV file
        df.to_csv(os.path.join(self.dataset_dir, "labels.csv"), index=False)

        print("Images and annotations files saved!")


# For debugging
if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = FashionMNISTDataset(train=True, transform=transform)
    # print(len(train_dataset))
    print(train_dataset[0][0].size())

    test_dataset = FashionMNISTDataset(train=False, transform=transform)
    # print(len(test_dataset))
    # print(test_dataset[0])
