import torch

from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset
import numpy as np
import torch.nn as nn
import os
import torch.optim as optim
import boto3
from sagemaker.s3 import S3Uploader, S3Downloader
import argparse
import re
from torchvision import datasets, transforms
import json
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
from collections import Counter
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import s3fs
import random
from torch.utils.data import random_split

s3_client = boto3.client('s3')
os.environ['AWS_DEFAULT_REGION'] = 'eu-central-1'
session = boto3.Session(region_name=os.environ['AWS_DEFAULT_REGION'])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

def load_shadow_models(args):
    """
    Downloads the shadow models from S3 and loads them into memory.
    """
    shadow_models = []
    local_dir = "./shadow_models"
    os.makedirs(local_dir, exist_ok=True)

    clean_prefix = re.sub(r"^s3://[^/]+/", "", args.s3_shadow_prefix).rstrip('/')

    for i in range(1, args.num_shadow_models + 1):
        model_s3_key = f"{clean_prefix}/checkpoint_shadow_{i}.pth"
        local_path = os.path.join(local_dir, f"checkpoint_shadow_{i}.pth")

        #print(f"Downloading {model_s3_key} from S3 bucket {args.s3_bucket} to {local_path}...")
        
        # Use boto3 to download the exact file
        s3_client.download_file(args.s3_bucket, model_s3_key, local_path)

        # Verify the file was downloaded
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Model file {local_path} was not downloaded correctly.")

        model = TissueMNISTShadowModel()
        checkpoint = torch.load(local_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        shadow_models.append(model)

    return shadow_models


def load_datasets_from_s3(args, dataset_prefix, local_dir="./datasets"):
    """
    Loads all dataset splits from an S3 bucket and saves them locally.
    
    Args:
        bucket_name (str): Name of the S3 bucket.
        dataset_prefix (str): Prefix in S3 where dataset splits are stored.
        local_dir (str): Local directory to store the datasets.

    Returns:
        dict: Dictionary containing loaded datasets with keys as split names.
    """
    os.makedirs(local_dir, exist_ok=True)

    # List dataset splits
    response = s3_client.list_objects_v2(Bucket=args.s3_bucket, Prefix=dataset_prefix)
    
    if "Contents" not in response:
        raise ValueError("No datasets found in the specified S3 path.")

    datasets_dict = {}

    for obj in response["Contents"]:
        key = obj["Key"]
        if key.endswith(".json"):  # Ignore JSON mapping file for now
            continue

        split_name = key.split("/")[-2]  # Extract split_1, split_2, etc.
        local_file_path = os.path.join(local_dir, key.split("/")[-1])

        # Download dataset file
        s3_client.download_file(args.s3_bucket, key, local_file_path)

        # Load dataset
        transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
        dataset = datasets.ImageFolder(os.path.dirname(local_file_path), transform=transform)

        datasets_dict[split_name] = dataset

    return datasets_dict


def generate_attack_data(args, shadow_models, datasets):
    """
    Generate Attack adataset for each class
    Returns: 
        Attack_datasets(dict) :a dictionary of datasets for each class 
        (Binary Attack datasets)
    """
    print("I am generating attack data")
    attack_datasets = {cls: {'members': [], 'non_members': []}
                       for cls in range(args.num_classes)}

    for cls in range(args.num_classes):

        for model, dataset in zip(shadow_models, datasets):
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            model.eval()
            model.to("cuda")
            for inputs, labels in dataloader:
                with torch.no_grad():
                    inputs, labels = inputs.to("cuda"), labels.to("cuda")
                    outputs = model(inputs)
                    probs = outputs.cpu()#torch.softmax(outputs, dim=1).cpu()
                    mask = labels == cls
                    mask_exclude = labels != cls
                    kept = probs[mask]
                    non_kept = probs[mask_exclude]
                    if kept.shape[0] > 0:
                        confidence_vect = torch.chunk(kept, kept.shape[0], dim=0)
                        #members and non_members
                        for el in confidence_vect:
                            attack_datasets[cls]['members'].append(el)
                    if non_kept.shape[0] >0:
                        confidence_vect_excluded_cls = torch.chunk(non_kept, non_kept.shape[0], dim=0)
                        for e in confidence_vect_excluded_cls:
                             attack_datasets[cls]['non_members'].append(e)
    return attack_datasets

class AttackModel2(nn.Module):
    def __init__(self, input_size):
        super(AttackModel2, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.output(x))
        return x
    
def prepare_binary_attack_datasets_(attack_datasets):
    print("I am preparing the PyTorch dataset")
    binary_datasets = []

    for cls in attack_datasets:
        members = np.array([np.array(m) for m in attack_datasets[cls]["members"]])
        non_members = np.array([np.array(nm) for nm in attack_datasets[cls]["non_members"]])

        print(f"Class {cls}:")
        print(f"  Members: {len(members)}, Shape: {members.shape if len(members) > 0 else 'Empty'}")
        print(f"  Non-members: {len(non_members)}, Shape: {non_members.shape if len(non_members) > 0 else 'Empty'}")

        if len(members) == 0 or len(non_members) == 0:
            print(f"Skipping class {cls} due to missing data.")
            continue  # Avoid errors by skipping empty datasets

        try:
            data = np.row_stack((members, non_members))  # Stack along the first axis
            labels = np.hstack((np.ones(len(members)), np.zeros(len(non_members))))

            binary_datasets.append(TensorDataset(torch.tensor(data, dtype=torch.float32),
                                                 torch.tensor(labels, dtype=torch.float32)))
        except Exception as e:
            print(f"Error processing class {cls}: {e}")

    return binary_datasets


def prepare_binary_attack_datasets(attack_datasets):
    print("Preparing balanced PyTorch datasets")
    binary_datasets = []

    for cls in attack_datasets:
        members = [np.array(m) for m in attack_datasets[cls]["members"]]
        non_members = [np.array(nm) for nm in attack_datasets[cls]["non_members"]]

        print(f"Class {cls}: Members={len(members)}, Non-members={len(non_members)}")

        if len(members) == 0 or len(non_members) == 0:
            print(f"Skipping class {cls} due to missing data.")
            continue  # Skip if one of the classes is empty

        # **Balance the dataset**
        min_size = min(len(members), len(non_members))  # Find the smallest count
        members = random.sample(members, min_size)  # Downsample members
        non_members = random.sample(non_members, min_size)  # Downsample non-members

        # Convert to numpy arrays
        members = np.array(members)
        non_members = np.array(non_members)

        # Stack data and labels
        data = np.vstack((members, non_members))
        labels = np.hstack((np.ones(len(members)), np.zeros(len(non_members))))

        # Create PyTorch dataset
        binary_datasets.append(TensorDataset(torch.tensor(data, dtype=torch.float32),
                                             torch.tensor(labels, dtype=torch.float32)))

        print(f"Balanced dataset for class {cls}: {len(members)} members, {len(non_members)} non-members")

    return binary_datasets

def save_binary_datasets(args,binary_datasets):
    """
    Saves the binary datasets to the specified directory and uploads them to the specified S3 bucket path.

    Args:
        binary_datasets (list): A list of binary datasets to be saved

    Returns:
        None
    """
    os.makedirs("./Binary_datasets", exist_ok=True)
    for i, dataset in enumerate(binary_datasets):
        file_path = f"./Binary_datasets/binary_dataset_{i}.pt"
        torch.save(dataset, file_path)
        S3Uploader.upload(file_path, f"s3://{args.s3_bucket}/{args.s3_binary_dataset_path}/binary_dataset_{i}.pt")



def split_dataset(dataset, val_ratio=0.2):
    """
    Splits a dataset into training and validation subsets.
    
    Args:
        dataset (Dataset): The dataset to split.
        val_ratio (float): The fraction of data to use for validation.
    
    Returns:
        Tuple[Dataset, Dataset]: Training and validation datasets.
    """
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset
    
def train_attack_models(args, binary_datasets):
    """
    Train an attack model for each class in the given binary datasets.

    Args:
        args (argparse.Namespace): The parsed arguments containing hyperparameters.
        binary_datasets (list): A list of binary datasets for training attack models.

    Returns:
        list: A list of trained attack models.
    """
    attack_models = []
    
    print("Let's train our attack models!")
    os.makedirs("/tmp/attack_models", exist_ok=True)

    for cls, dataset in enumerate(binary_datasets):
        print(f"Training attack model for class {cls}...")

        # Split dataset into training and validation
        train_dataset, val_dataset = split_dataset(dataset, val_ratio=0.2)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # Initialize attack model

        model = AttackModel2(input_size=args.input_size).to("cuda")
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train()

        # Store training history for plotting
        history = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc_roc': []}

        for epoch in range(args.epochs):
            running_loss = 0.0

            for inputs, labels in train_loader:
                print(f" for debugging the input size for the model is {args.input_size}")
                print(f" for debugging the input size for the data is {inputs.size()}")

                inputs, labels = inputs.to("cuda"), labels.to("cuda")
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels.float())  # Ensure labels are float for BCELoss
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Evaluate on validation set
            acc, prec, rec, f1, auc = evaluate_attack_model(model, val_loader, device)

            # Store metrics
            history['accuracy'].append(acc)
            history['precision'].append(prec)
            history['recall'].append(rec)
            history['f1'].append(f1)
            history['auc_roc'].append(auc)

            avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
            print(f"Epoch {epoch + 1}, Loss: {avg_loss}, Acc: {acc:.4f}, AUC: {auc:.4f}")

        # Plot performance after training
        plot_attack_performance(history, class_label=cls)

        # Save model
        model_path = f"/tmp/attack_models/attack_model_{cls}.pth"
        torch.save(model.state_dict(), model_path)
        attack_models.append(model)

        # Upload to S3
        s3_client.upload_file(model_path, args.s3_bucket, f"attack_models/attack_model_{cls}.pth")

    return attack_models

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
        #print("Raw class folders from S3:", class_folders)  # Debugging output
    
        for class_folder in class_folders:
            folder_name = os.path.basename(class_folder)
            print(f"Extracted folder name: {folder_name}")  # Debugging output
    
            # Extract class label from [0], [1], etc.
            match = match = re.search(r"\[?(\d+)\]?", folder_name)

            if not match:
                print(f"Skipping invalid folder: {folder_name}")  # Debugging output
                continue
            label = int(match.group(1))
            #print(f"Extracted label: {label}")  # Debugging output
    
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
            #print(f"Found local folder: {class_folder}")  # Debugging output
            
            match = re.search(r"\[?(\d+)\]?",class_folder)

            if not match:
                #print(f"Skipping invalid folder: {class_folder}")  # Debugging output
                continue
            label = int(match.group(1))
            #print(f"Extracted label: {label}")  # Debugging output
    
            class_path = os.path.join(self.local_dir, class_folder)
            image_files = os.listdir(class_path)
            #print(f"Images in {class_folder}: {image_files}")  # Debugging output
    
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
    
def load_fashionProd_from_s3(args, local_dir="./datasets"):
    """
    Loads all Tiny ImageNet dataset splits from an S3 bucket and saves them locally.

    Args:
        args: Arguments containing S3 details.
            - args.s3_bucket (str): Name of the S3 bucket.
            - args.s3_dataset_prefix (str): Prefix in S3 where dataset splits are stored.
        local_dir (str): Local directory to store the datasets.

    Returns:
        dict: Dictionary containing TinyImageNetDataset instances for each split.
    """

    os.makedirs(local_dir, exist_ok=True)
    s3_client = boto3.client("s3")

    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize to match model input size
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    datasets_dict = {}

    # Function to list all S3 objects (handling pagination)
    def list_all_objects(bucket, prefix):
        """ Retrieves all objects from an S3 bucket with pagination handling. """
        objects = []
        continuation_token = None
        while True:
            list_kwargs = {'Bucket': bucket, 'Prefix': prefix}
            if continuation_token:
                list_kwargs['ContinuationToken'] = continuation_token

            response = s3_client.list_objects_v2(**list_kwargs)
            if "Contents" in response:
                objects.extend(response["Contents"])

            if response.get('IsTruncated'):  # More objects exist
                continuation_token = response['NextContinuationToken']
            else:
                break
        return objects

    # Retrieve all dataset objects
    all_objects = list_all_objects(args.s3_bucket, args.s3_dataset_prefix)

    if not all_objects:
        raise ValueError(f"No datasets found in the specified S3 path: s3://{args.s3_bucket}/{args.s3_dataset_prefix}")

    # Extract all unique split names
    split_names = set()
    for obj in all_objects:
        key = obj["Key"]
        if key.endswith(".json"):  # Skip JSON files
            continue
        parts = key.split("/")
        if len(parts) >= 3:
            split_names.add(parts[2])  # Extract "split_1", "split_2", ..., "split_5"

    print(f"Found dataset splits: {split_names}")

    # Process each split separately
    for split_name in sorted(split_names):  # Sorting ensures consistency
        split_prefix = f"{args.s3_dataset_prefix}/{split_name}/"
        local_split_path = os.path.join(local_dir, split_name)
        os.makedirs(local_split_path, exist_ok=True)

        # Get all objects within the split
        split_objects = [obj for obj in all_objects if obj["Key"].startswith(split_prefix)]

        if not split_objects:
            print(f"Warning: No files found in {split_name}")
            continue

        print(f"Downloading files for {split_name}... ({len(split_objects)} files)")

        for obj in split_objects:
            key = obj["Key"]
            if key.endswith(".json"):
                continue  # Skip JSON files

            relative_path = key[len(args.s3_dataset_prefix) + 1:]  # Keep directory structure
            local_file_path = os.path.join(local_dir, relative_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Download the file
            s3_client.download_file(args.s3_bucket, key, local_file_path)

        # Debug: Check how many files are inside the split directory
        num_files = sum([len(files) for _, _, files in os.walk(local_split_path)])
        print(f"Downloaded {num_files} files for {split_name}.")

        if num_files == 0:
            print(f"Error: No images found in {split_name} after download!")

        # Add dataset instance
        datasets_dict[split_name] = TissueMNISTDatasetOptimized(args.s3_bucket, split=split_name, transform=transform)

    return datasets_dict



def evaluate_attack_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).view(-1)  # Ensures outputs is 1D
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int).reshape(-1)  # Ensures preds is 1D
            all_preds.extend(preds.tolist())  # Convert NumPy array to list before extending
            all_labels.extend(labels.cpu().numpy().astype(int).tolist())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")

    # Debugging: Print class distribution
    label_counts = Counter(all_labels)
    print(f"Class distribution in validation set: {label_counts}")

    # Handle AUC computation safely
    if len(set(all_labels)) < 2:  # Only one class present
        warnings.warn("Only one class present in y_true. ROC AUC score is not defined.", UndefinedMetricWarning)
        auc_roc = float("nan")  # Set AUC to NaN or another placeholder
    else:
        auc_roc = roc_auc_score(all_labels, all_preds)

    return accuracy, precision, recall, f1, auc_roc


def plot_attack_performance(history, class_label):
    epochs = range(1, len(history['accuracy']) + 1)
    
    plt.figure(figsize=(10, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['accuracy'], label="Accuracy", marker="o")
    plt.plot(epochs, history['auc_roc'], label="AUC-ROC", marker="s")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title(f"Attack Model Performance (Class {class_label})")
    plt.legend()
    
    # Precision, Recall, F1 plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['precision'], label="Precision", marker="o")
    plt.plot(epochs, history['recall'], label="Recall", marker="s")
    plt.plot(epochs, history['f1'], label="F1-score", marker="d")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()
    
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--input_size', type=int, required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--num_shadow_models', type=int, required=True)
    parser.add_argument('--s3_binary_dataset_path', type=str, required=True)
    parser.add_argument('--s3_bucket', type=str, required=True)
    parser.add_argument('--s3_dataset_prefix', type=str, required=True)
    parser.add_argument('--s3_output_path', type=str, required=True)
    parser.add_argument('--s3_shadow_prefix', type=str, required=True)


    args = parser.parse_args()
    
    shadow_models = load_shadow_models(args)
    datasets = load_fashionProd_from_s3(args)
    datasets2 = list(datasets.values())
    non_members = [datasets2[-1]]
    members = datasets2[:-1]
    print("printing the number of datasets loaded ")
    print(len(  datasets2))
    print(len(  datasets2[0]))

    attack_datasets = generate_attack_data(args, shadow_models, datasets2)
    binary_datasets = prepare_binary_attack_datasets(attack_datasets)
    save_binary_datasets(args, binary_datasets)
    train_attack_models(args, binary_datasets)
    print("Training complete. Models and datasets saved to S3.")