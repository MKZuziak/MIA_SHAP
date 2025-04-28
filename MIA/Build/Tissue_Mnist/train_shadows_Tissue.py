import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import argparse
import sagemaker
from sagemaker.pytorch import PyTorchModel
import boto3
import json
from urllib.parse import urlparse
from botocore.exceptions import NoCredentialsError
import torch.nn.functional as F
import s3fs
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

import re


s3 = boto3.client("s3")

class TissueMNISTShadowModel(nn.Module):
    def __init__(self):
        super(TissueMNISTShadowModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Output: 28x28x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: 28x28x64
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 14x14x64
        
        # Additional Convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output: 14x14x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 7x7x128
        
        # Fully connected layers
        self.fc1 = nn.Linear(7 * 7 * 128, 256)
        self.fc2 = nn.Linear(256, 8)  # 8 classes in TissueMNIST
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # First pooling
        
        x = F.relu(self.conv3(x))
        x = self.pool2(x)  # Second pooling
        
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))  # Apply dropout
        x = self.fc2(x)
        
        return x




class TissueMNISTDatasetOptimized(Dataset):
    def __init__(self, s3_bucket, split, local_dir="/mnt/data/tissuemnist/", transform=None):
        self.s3 = s3fs.S3FileSystem()
        self.s3_bucket = s3_bucket
        self.split_path = f"{s3_bucket}/datasets/tissuemnist-splits/{split}/"
        self.local_dir = os.path.join(local_dir, split)
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Download dataset to local storage if not already cached
        if not os.path.exists(self.local_dir):
            os.makedirs(self.local_dir, exist_ok=True)
            self._download_data()

        # Load images from local storage
        self._load_local_data()

    def _download_data(self):
        """Download all images from S3 to local storage."""
        print(f"Downloading dataset from S3: {self.split_path}")
        
        class_folders = self.s3.ls(self.split_path)
        print("Raw class folders from S3:", class_folders)  # Debugging output
    
        for class_folder in class_folders:
            folder_name = os.path.basename(class_folder)
            print(f"Extracted folder name: {folder_name}")  # Debugging output
    
            # Extract class label from [0], [1], etc.
            match = match = re.search(r"\[?(\d+)\]?", folder_name)

            if not match:
                print(f"Skipping invalid folder: {folder_name}")  # Debugging output
                continue
            label = int(match.group(1))
            print(f"Extracted label: {label}")  # Debugging output
    
            local_class_dir = os.path.join(self.local_dir, folder_name)
            os.makedirs(local_class_dir, exist_ok=True)
    
            image_files = self.s3.ls(class_folder)
            #print(f"Images in {folder_name}: {image_files}")  # Debugging output
            
            for image_file in image_files:
                if image_file.endswith(('.png', '.jpg', '.jpeg')):
                    local_file = os.path.join(local_class_dir, os.path.basename(image_file))
                    if not os.path.exists(local_file):  # Avoid re-downloading
                        #print(f"Downloading {image_file} to {local_file}")  # Debugging output
                        self.s3.get(image_file, local_file)

    def _load_local_data(self):
        """Load dataset from local storage after download."""
        print(f"Loading dataset from local storage: {self.local_dir}")
    
        for class_folder in sorted(os.listdir(self.local_dir)):
            print(f"Found local folder: {class_folder}")  # Debugging output
            
            match = re.search(r"\[?(\d+)\]?",class_folder)

            if not match:
                print(f"Skipping invalid folder: {class_folder}")  # Debugging output
                continue
            label = int(match.group(1))
            print(f"Extracted label: {label}")  # Debugging output
    
            class_path = os.path.join(self.local_dir, class_folder)
            image_files = os.listdir(class_path)
            print(f"Images in {class_folder}: {image_files}")  # Debugging output
    
            for image_file in image_files:
                if image_file.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_path, image_file))
                    self.labels.append(label)
                    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, label




def save_checkpoint(model, loss, filename):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }
    # Create a local directory to temporarily save checkpoints before uploading to S3
    local_checkpoint_dir = '/opt/ml/model/checkpoints'
    os.makedirs(local_checkpoint_dir, exist_ok=True)

    # Save the checkpoint locally first
    checkpoint_filename = os.path.join(local_checkpoint_dir, filename)
    torch.save(checkpoint, checkpoint_filename)
    
    # Now upload the checkpoint to S3
    try:
        s3.upload_file(checkpoint_filename, "   YOUR BUCKET", f'checkpoints/{filename}')
        print(f"Checkpoint for shadow  saved to S3.")
    except NoCredentialsError:
        print("Credentials not available.")
    except Exception as e:
        print(f"Error uploading checkpoint to S3: {e}")

# Training function
def train_model(args):

    # Setup transformations
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Dataset setup
    split_name = f"split_{args.number}"
    dataset = TissueMNISTDatasetOptimized(s3_bucket= args.bucket_name,split=split_name, transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Model setup
    model = TissueMNISTShadowModel().to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            #print(f"Batch size (inputs): {inputs.shape}, Batch size (labels): {labels.shape}")

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # Save model
    checkpoint_filename = f"checkpoint_shadow_{str(args.number)}.pth"
    save_checkpoint(model, running_loss, checkpoint_filename)


def download_s3_file(s3_uri, local_path):
    # Ensure the input is an actual S3 URI
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")

    parts = s3_uri.replace("s3://", "").split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 URI format: {s3_uri}")

    bucket_name, key = parts
    s3 = boto3.client("s3")
    s3.download_file(bucket_name, key, local_path)

    
# Update this section in `train.py`
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/opt/ml/input/data/training")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model_dir", type=str, default="/opt/ml/model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--number', type=int, required=True, help='Split number (1-5) for shadow models')
    parser.add_argument('--bucket_name', type=str, required=True)
    args = parser.parse_args()

    train_model(args)
   
    
    
