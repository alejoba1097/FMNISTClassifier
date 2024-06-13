from src.dataset import FashionMNISTDataset
from src.model import FashionMNISTClassifier
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import os
import json
from typing import Tuple


def train(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: optim,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """
    Training loop function to perform a full pass for each batch,
    compute the losses and update the model weights.

    Args:
        model (nn.Module): Pytorch model to be trained
        device (torch.device): Device to use for the training
        train_loader (DataLoader): DataLoader instance to loop through the training set
        optimizer (optim): Optimizer to be used to update the weights
        criterion (nn.Module): Loss function to compute the gradient

    Returns:
        Tuple[float, float]:
            - avg_loss: Average training loss computed over the batches
            - accuracy: Accuracy computed over the whole dataset
    """
    # Set model to training state
    model.train()

    # Initiliaze accumulators
    total_loss = 0
    correct = 0

    # DataLoader loop to iterate through each batch
    for batch, (images, labels) in enumerate(train_loader):
        # Forward pass
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        # Backward pass and weights update
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Compute loss and accuracy for current batch
        total_loss += loss.item()

        predicted_labels = outputs.argmax(dim=1)
        correct += predicted_labels.eq(labels.view_as(predicted_labels)).sum().item()

    # Compute loss and accuracy across all batches
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / len(train_loader.dataset)

    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    device: torch.device,
    data_loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """
    Evaluate function to perform a full pass for each image
    in validation and test.

    Args:
        model (nn.Module): Model to be used for the inference.
        device (torch.device): Device to use for the training.
        data_loader (DataLoader): DataLoader instance to loop through the dataset.
        criterion (nn.Module): Loss function to be computed.

    Returns:
        Tuple[float, float]:
            - avg_loss: Average training loss computed over the batches.
            - accuracy: Accuracy computed over the whole dataset.
    """

    # Set model to evaluating state
    model.eval()

    # Initiliaze accumulators
    total_loss = 0
    correct = 0

    # Ensure no gradient is computed
    with torch.no_grad():
        # DataLoader loop to iterate through each batch
        for images, labels in data_loader:
            # Forward pass
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Compute loss and accuracy for current batch
            total_loss += criterion(outputs, labels).item()

            predicted_labels = outputs.argmax(dim=1)
            correct += (
                predicted_labels.eq(labels.view_as(predicted_labels)).sum().item()
            )

        avg_loss = total_loss / len(data_loader)
        accuracy = correct / len(data_loader.dataset)

    return avg_loss, accuracy


def main():
    """
    Main function to configure the training, MLflow and model configuration,
    and calling training and validation loops.
    """
    print("Loading dataset")

    # Load configuration file
    with open("model/config.json", "r") as config_file:
        config = json.load(config_file)

    # Load training parameters
    learning_rate, batch_size, max_epochs, train_val_ratio = config["training"].values()

    # Set device to cuda in case it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformations to be applied to each image when read
    transform = transforms.Compose([transforms.ToTensor()])

    # Create dataset object and generate the training and validation split
    train_set, validation_set = random_split(
        FashionMNISTDataset(transform=transform), [train_val_ratio, 1 - train_val_ratio]
    )

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size)
    validation_loader = DataLoader(validation_set, batch_size=1000)

    # Instanciate the model and move it to the device
    model = FashionMNISTClassifier().to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load MLflow configuration
    remote_server_uri, experiment_name = config["mlflow"].values()

    # Set MLflow configuration
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)

    # Start MLflow run for logging capabilities
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(config["training"])

        # Training loop
        print("Training loop started")

        for epoch in range(max_epochs):
            print("Starting epoch: " + str(epoch))
            # Training step
            train_loss, train_accuracy = train(
                model, device, train_loader, optimizer, criterion
            )

            # Validation step
            validation_loss, validation_accuracy = evaluate(
                model, device, validation_loader, criterion
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                },
                step=epoch,
            )

            # Implement chekpointing
            if epoch % 5 == 0:
                checkpoint_path = f"mlflow/checkpoint_epoch_{epoch}.pth"
                torch.save(model.state_dict(), checkpoint_path)
                mlflow.log_artifact(checkpoint_path)
                os.remove(checkpoint_path)

            # Optional: print results from the training loop
            print(f"Finisehd epoch {epoch}")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            print(
                f"Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.4f}"
            )

        mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    main()
