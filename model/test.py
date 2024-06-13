from src.dataset import FashionMNISTDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import mlflow
import json
from typing import Tuple

# Set device to cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([transforms.ToTensor()])

# Create Dataset and DataLoadet
test_set = FashionMNISTDataset(transform=transform)
test_loader = DataLoader(test_set, batch_size=1000)

# Load configuration file
with open("model/config.json", "r") as config_file:
    config = json.load(config_file)

# Configure MLflow
remote_server_uri = config["mlflow"]["remote_server_uri"]
model_uri = config["test"]["model_uri"]
mlflow.set_tracking_uri(remote_server_uri)
model = mlflow.pytorch.load_model(model_uri)

# Define loss function
criterion = nn.CrossEntropyLoss()


def evaluate(
    model: nn.Module,
    device: torch.device,
    data_loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[float, float]:
    # Set model to evaluation state
    model.eval()

    # Initialize accumulators
    total_loss = 0
    correct = 0

    # Ensure no gradient is computed
    with torch.no_grad():
        # Loop through the test dataset
        for images, labels in data_loader:
            # Forward pass
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Compute loss and accuracy for the batch
            total_loss += criterion(outputs, labels).item()

            predicted_labels = outputs.argmax(dim=1)
            correct += (
                predicted_labels.eq(labels.view_as(predicted_labels)).sum().item()
            )

        # Compute average loss and accuracy
        avg_loss = total_loss / len(data_loader)
        accuracy = correct / len(data_loader.dataset)

    return avg_loss, accuracy


if __name__ == "__main__":
    test_loss, test_accuracy = evaluate(model, device, test_loader, criterion)

    print(f"Test loss: {test_loss}. Test accuracy: {test_accuracy}")
