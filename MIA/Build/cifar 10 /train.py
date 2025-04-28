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

s3 = boto3.client("s3")


# Define the model
class ShadowCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ShadowCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x =  x.view(-1, 128 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TinyImageNetDataset(Dataset):
    def __init__(self, data_dir, class_mapping, transform=None):
        """
        Custom Dataset for loading Tiny ImageNet.
        
        :param data_dir: Directory containing images.
        :param class_mapping: Mapping from class folder name to class ID.
        :param transform: Transformations to apply to the images.
        """
        self.data_dir = data_dir
        self.class_mapping = class_mapping
        self.transform = transform
        self.image_paths = []
        
        # Load image paths
        for class_name in os.listdir(data_dir):
            class_folder = os.path.join(data_dir, class_name)
            if os.path.isdir(class_folder):
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    self.image_paths.append((img_path, class_name))
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path, class_name = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.class_mapping[class_name]
        
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
        s3.upload_file(checkpoint_filename, "YOUR BUCKET", f'checkpoints/{filename}')
        print(f"Checkpoint for shadow  saved to S3.")
    except NoCredentialsError:
        print("Credentials not available.")
    except Exception as e:
        print(f"Error uploading checkpoint to S3: {e}")

# Training function
def train_model(args):
    # Load class mapping from JSON
    
    local_class_mapping_path = "/opt/ml/input/class_mapping.json"
    download_s3_file(args.class_mapping, local_class_mapping_path)
    
    with open(local_class_mapping_path) as f:
        class_mapping = json.load(f)

    # Setup transformations
    transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Explicitly set to 64x64
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Dataset setup
    dataset = TinyImageNetDataset(data_dir=args.data_dir, class_mapping=class_mapping, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model setup
    model = ShadowCNN(num_classes=len(class_mapping)).to(args.device)
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
            print(f"Batch size (inputs): {inputs.shape}, Batch size (labels): {labels.shape}")

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
    parser.add_argument("--class_mapping", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model_dir", type=str, default="/opt/ml/model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--number', type=int, required=True, help='Split number (1-5) for shadow models')

    args = parser.parse_args()

    # Fix: Download `class_mapping2.json` from S3 to a local file
    local_class_mapping_path = "/opt/ml/input/data/class_mapping2.json"
    download_s3_file(args.class_mapping, local_class_mapping_path)

    # Load class mapping from the downloaded file
    with open(local_class_mapping_path) as f:
        class_mapping = json.load(f)

    train_model(args)
