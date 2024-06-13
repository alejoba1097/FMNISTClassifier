import torch
import torch.nn as nn
from torch import Tensor
from torchsummary import summary


class FashionMNISTClassifier(nn.Module):
    """
    Class for the Fashion-MNIST classifier model.

    Extends from torch.nn.Module.
    """

    def __init__(self):
        """
        Initialization function for the classifier model.

        Defines the following layers:
        - conv1: Convolutional network that maps the 1x28x28 image to 32x14x14 features
        - conv2: Convolutional network that maps the 32x7x7 features to 64x7x7 features
        - pool: MaxPooling layer to extract the characteristics of each resulting feature
        - linear1: Fully connected linear layer to map down the resulting 64x7x7 features to 128
        - linear2: Fully connected linear layer to map the 128 features to the number of classes: 10
        """
        super(FashionMNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(64 * 7 * 7, 128)
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (Tensor): Input tensor with shape (batch_size, channels, height, width)

        Returns:
            Tensor: Output tensor with shape (batch_size, num_classes)
        """
        # Pass through convolutional and pooling layers
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))

        # Reshaping for fully connected layers
        x = x.view(-1, 64 * 7 * 7)

        # Pass through fully connected layers
        x = nn.functional.relu(self.linear1(x))
        x = self.linear2(x)

        return x


# For debugging
if __name__ == "__main__":
    model = FashionMNISTClassifier()
    print(summary(model, (1, 28, 28)))

    x = torch.rand((64, 1, 28, 28))
    print(model(x).size())
