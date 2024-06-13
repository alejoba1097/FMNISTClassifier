import mlflow
import uvicorn
import io
import json
import torch.nn as nn
from fastapi import FastAPI, UploadFile
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict


def get_model() -> nn.Module:
    """
    Main function to load configuration and model

    Returns:
        nn.Module: Model to be used for inference
        dict: Labels map from class number to text
    """

    # Load configuration file
    with open("config.json", "r") as config_file:
        config = json.load(config_file)

    # Load MLflow configuration
    remote_server_uri, model_uri = config["mlflow"].values()
    mlflow.set_tracking_uri(remote_server_uri)

    print(model_uri)

    # Load model and set status to evaluating
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()

    # Load labels map
    labels_map = config["labels_map"]

    return model, labels_map


# Initialize model and labels map
model, labels_map = get_model()

# Initialize FastAPI app
fmnist_app = FastAPI()


@fmnist_app.post("/predict")
def predict(image: UploadFile) -> Dict:
    """
    Returns a dictionary containing the predicted label for an image.
    Uses POST method to be called.

    Args:
        image (UploadFile): PNG image provided to the API.

    Returns:
        Dict: Prediction results from the model
    """

    # Load image using PIL
    image = Image.open(io.BytesIO(image.file.read()))
    image = image.convert("L")

    # Transform image to tensor
    transform = transforms.Compose([transforms.ToTensor()])

    # Add batch dimension for compatibility
    image = transform(image).unsqueeze(0)

    # Forward pass
    outputs = model(image)
    label_str = outputs.argmax(dim=1).item()

    # Label mapping
    label_name = labels_map[str(label_str)]

    return {"label_number": str(label_str), "article_name": label_name}


if __name__ == "__main__":
    uvicorn.run(fmnist_app, host="0.0.0.0", port=8080)
