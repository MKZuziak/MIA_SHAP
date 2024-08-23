import torch
import torch.nn.functional as F
from torch import Tensor, nn

torch.manual_seed(42)


class MNIST_MLP(nn.Module):
    """Mnist network definition."""

    def __init__(self) -> None:
        """Initialization of the network."""
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(
                in_features= 784,
                out_features= 100
            ),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(
                in_features=100,
                out_features=50
            ),
            nn.ReLU()
        )
        self.out = nn.Linear(in_features=50,
                             out_features=10)

    def forward(self, input_data: Tensor) -> Tensor:
        """Defines the forward pass of the network.
        Args:
            input_data (Tensor): Input data
        Returns
        -------
            Tensor: Output data
        """
        x = input_data.view(-1, 784)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x


class MNIST_Dropout_MLP(nn.Module):
    """Mnist network definition."""

    def __init__(self) -> None:
        """Initialization of the network."""
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(28 * 28, 50),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(50, 250),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(250, 50),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fc4 = nn.Linear(50, 10)

    def forward(self, input_data: Tensor) -> Tensor:
        """Defines the forward pass of the network.
        Args:
            input_data (Tensor): Input data
        Returns
        -------
            Tensor: Output data
        """
        x = input_data.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return self.fc4(x)


class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()      
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),    
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )           
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 7 * 7, 250),
            nn.ReLU(),
            nn.Dropout(0.2))
        self.fc2 = nn.Sequential(
            nn.Linear(250, 100),
            nn.ReLU(),
            nn.Dropout(0.2))
        self.fc3 = nn.Linear(100, 10)    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 64 * 7 * 7) 
        x = self.fc1(x)
        x = self.fc2(x)      
        x = self.fc3(x)
        return x


class MNIST_Expanded_CNN(nn.Module):
    def __init__(self):
        super().__init__()      
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),    
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2
            ),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2
            ),
            nn.ReLU()
        )           
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 28 * 28, 1000),
            nn.ReLU(),
            nn.Dropout(0.2))
        self.fc2 = nn.Sequential(
            nn.Linear(1000, 250),
            nn.ReLU(),
            nn.Dropout(0.2))
        self.fc3 = nn.Sequential(
            nn.Linear(250, 100),
            nn.ReLU(),
            nn.Dropout(0.2))
        self.fc4 = nn.Linear(100, 10)    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 32 * 28 * 28) 
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x