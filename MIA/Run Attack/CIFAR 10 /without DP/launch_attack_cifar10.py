import torch
import os
import json
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import random

import boto3
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
# Install timm if not available


from datasets import Dataset as HFDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,  classification_report, roc_auc_score, confusion_matrix
)
import torch.nn as nn
from collections import defaultdict
import re
import argparse
import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "-q","install", package])

install("datasets")

from datasets import concatenate_datasets





# Custom Dataset for Hugging Face Arrow Data
class ArrowToTorchDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset
        self.transform = ToTensor()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = self.transform(item["image"])  # Ensure only ToTensor()
        label = torch.tensor(item["label"]).long()
        return image, label

class  Attack_model(nn.Module):
    def __init__(self, input_size):
        super(Attack_model, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(1)  # Remove extra dimension

        return self.fc(x)

def download_arrow_files_from_s3(args, prefix, local_dir="./"):
    """
    Searches for .arrow files in the specified S3 path and downloads them to a local directory.
    
    Args:
        bucket_name (str): Name of the S3 bucket.
        s3_prefix (str): S3 prefix (folder path) where the .arrow files are stored.
        local_dir (str): Local directory to save the downloaded files.
    
    Returns:
        List of downloaded file paths.
    """
    s3_client = boto3.client("s3")
    
    # Ensure the local directory exists
    os.makedirs(local_dir, exist_ok=True)
    
    # List all objects in the given S3 prefix
    response = s3_client.list_objects_v2(Bucket=args.dataset_bucket, Prefix=prefix)

    if "Contents" not in response:
        print(f"No files found in {prefix}.")
        return []

    downloaded_files = []

    for obj in response["Contents"]:
        file_key = obj["Key"]
        if file_key.endswith(".arrow"):
            local_file_path = os.path.join(local_dir, os.path.basename(file_key))
            print(f"Downloading {file_key} to {local_file_path}...")
            
            # Download the file
            s3_client.download_file(args.dataset_bucket, file_key, local_file_path)
            downloaded_files.append(local_file_path)
    
    return downloaded_files

def load_dataset(args, client_id, is_member, local_dir="./"):
    """
    Load dataset for a given client and membership status.

    Args:
        args: Arguments containing S3 bucket details.
        client_id (int): Client index (e.g., 0, 1, 2, ...).
        is_member (bool): True for member dataset, False for non-member dataset.
        local_dir (str): Local directory for storing downloaded files.

    Returns:
        Dataset (not DataLoader yet) for the given membership status.
    """
    if client_id == "orchestrator":
        if is_member:
            client_id = 1
            set_name = f"client_{client_id}_train_set"
        else:
            set_name = "orchestrator_test_set"

    else:
        set_name = f"client_{client_id}_{'train' if is_member else 'test'}_set"


    # Download dataset files
    files = download_arrow_files_from_s3(args, f"{args.dataset_prefix}{set_name}/")

    if not files:
        raise FileNotFoundError(f"No .arrow files found for {set_name} in {args.dataset_bucket}/{args.dataset_prefix}")

    # Load and concatenate multiple .arrow files if necessary
    datasets = [HFDataset.from_file(file) for file in files]
    hf_dataset = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]

    return ArrowToTorchDataset(hf_dataset)  # Return dataset, not DataLoader


def load_datasets(args, client_id, local_dir="./"):
    """
    Load and balance both member (train) and non-member (test) datasets for a given client.

    Args:
        args: Arguments containing S3 bucket details.
        client_id (int): Client index (e.g., 0, 1, 2, ...).
        local_dir (str): Local directory for storing downloaded files.

    Returns:
        Tuple of balanced DataLoaders: (balanced_train_loader, balanced_test_loader)
    """
    member_dataset = load_dataset(args, client_id, is_member=True, local_dir=local_dir)
    non_member_dataset = load_dataset(args, client_id, is_member=False, local_dir=local_dir)

    # Balance datasets by taking the smaller size
    min_size = min(len(member_dataset), len(non_member_dataset))

    # Randomly sample from larger dataset
    member_indices = random.sample(range(len(member_dataset)), min_size)
    non_member_indices = random.sample(range(len(non_member_dataset)), min_size)

    balanced_member_dataset = Subset(member_dataset, member_indices)
    balanced_non_member_dataset = Subset(non_member_dataset, non_member_indices)

    # Create DataLoaders with the balanced datasets
    balanced_train_loader = DataLoader(balanced_member_dataset, batch_size=args.batch_size, shuffle=True)
    balanced_test_loader = DataLoader(balanced_non_member_dataset, batch_size=args.batch_size, shuffle=True)

    return balanced_train_loader, balanced_test_loader

###############
    

def load_target(checkpoint_path):
    model = ResNet34(BasicBlock, num_classes=10)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    # If the model was saved using DataParallel, remove '_module.' prefix
    checkpoint = {key.replace('_module.', ''): value for key, value in checkpoint.items()}

     #Load the state dict into the model
    model.load_state_dict(checkpoint)
    model.eval()
    return model
     
########################################################################
# This is needed to avoid opacus 


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(32, planes)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(32, planes)
        self.act2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act2(out)

        return out

# Define the ResNet34 model
class ResNet34(nn.Module):
    def __init__(self, block, num_classes=10):
        super(ResNet34, self).__init__()
        
        self.inplanes = 64  # Initialize inplanes here (number of input channels to the first layer)

        # Change input channels to 3 (CIFAR-10 is RGB)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.GroupNorm(32, self.inplanes)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Creating layers with block
        self.layer1 = self._make_layer(block, 64, 3)  # 3 blocks in the first layer
        self.layer2 = self._make_layer(block, 128, 4, stride=2)  # 4 blocks in the second layer
        self.layer3 = self._make_layer(block, 256, 6, stride=2)  # 6 blocks in the third layer
        self.layer4 = self._make_layer(block, 512, 3, stride=2)  # 3 blocks in the fourth layer

        # Global Average Pooling and Fully Connected layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # If stride is not 1 or the number of input channels doesn't match the output, downsample is required
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(32, planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion  # Update inplanes after the block
        for _ in range(1, blocks):  # Add remaining blocks
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Pass through initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        # Pass through all layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global Average Pooling and Fully Connected layer
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x





# This is needed to avoid opacus  
########################################################################
 


    
    
 
def load_attack(attack_model_path):
    model = Attack_model(input_size=10)
    model.load_state_dict(torch.load(attack_model_path,map_location=torch.device("cpu")))
    model.eval()
    return model



#Perform inference to get logits
def get_logits(model, dataloader):
    model.eval()
    logits_list, labels_list = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
           # inputs = inputs.to(next(model.parameters()).device)
            logits = model(inputs).cpu()  # Keep as torch.Tensor
            logits = F.softmax(logits, dim=1)
            logits_list.append(logits)
            labels_list.append(labels.cpu())

    return torch.cat(logits_list).float(), torch.cat(labels_list)


def download_attack_models_from_s3(args, local_dir="./attack_models"):
    """
    Downloads all attack models from S3 and returns a dictionary mapping class indices to local paths.

    Args:
        bucket_name (str): S3 bucket name.
        attack_model_prefix (str): S3 prefix where attack models are stored.
        local_dir (str): Local directory to store downloaded models.

    Returns:
        dict: Mapping of class indices (0-9) to local model paths.
    """
    os.makedirs(local_dir, exist_ok=True)
    s3 = boto3.client("s3")

    attack_model_paths = {}

    for cls in range(10):  # Assuming attack models are named as class_0.pt, class_1.pt, ...
        s3_key = f"{args.attack_model_prefix}/attack_model_{cls}.pth"
        print(f"Attempting to download from S3: {args.attack_bucket}/{s3_key}")
        local_path = os.path.join(local_dir, f"class_{cls}.pt")

        s3.download_file(args.attack_bucket, s3_key, local_path)
        attack_model_paths[cls] = local_path

    return attack_model_paths


def run_attack_local(args, target_model, attack_models, members_dataloader, nonmembers_dataloader, device="cpu"):
    """
    Runs the membership inference attack using class-specific attack models.

    Args:
        target_model (torch.nn.Module): The target model under attack.
        attack_model_paths (dict): Dictionary mapping class index to attack model paths.
        members_dataloader (DataLoader): Dataloader for members.
        nonmembers_dataloader (DataLoader): Dataloader for non-members.
        bucket_name (str): S3 bucket where attack models are stored.
        attack_model_prefix (str): Prefix for attack model paths in S3.
        device (str): Device to run inference on ("cpu" or "cuda").

    Returns:
        dict: Dictionary containing attack metrics per class.
    """
    target_model.to(device).eval()

    
    # Get logits for both member and non-member datasets
    member_logits, member_labels = get_logits(target_model, members_dataloader)
    nonmember_logits, nonmember_labels = get_logits(target_model, nonmembers_dataloader)

    # Store attack predictions
    attack_results = defaultdict(list)

    for cls in range(10):  # Iterate over all classes (0-9)
        attack_model = attack_models[cls]
        attack_model.eval()

        # Select samples where the target class matches cls
        member_mask = member_labels == cls
        nonmember_mask = nonmember_labels == cls

        if member_mask.sum() == 0 or nonmember_mask.sum() == 0:
            print(f"Skipping class {cls} due to no samples.")
            continue

        member_inputs = torch.tensor(member_logits[member_mask]).float().to(device)
        nonmember_inputs = torch.tensor(nonmember_logits[nonmember_mask]).float().to(device)
       
        # Predict membership for each class-specific attack model
        with torch.no_grad():
            member_preds = attack_model(member_inputs).cpu().numpy().squeeze()
            nonmember_preds = attack_model(nonmember_inputs).cpu().numpy().squeeze()
            
        attack_results[cls] = {
            "member_preds": member_preds,
            "nonmember_preds": nonmember_preds
        }

    return compute_attack_metrics(attack_results)



def compute_attack_metrics(attack_results):
    """
    Computes membership inference attack metrics.

    Args:
        attack_results (dict): Dictionary containing attack predictions for each class.

    Returns:
        dict: Metrics per class and averaged across all classes.
    """
    metrics = {}
    
    for cls, results in attack_results.items():
        member_preds = np.array(results["member_preds"])  # Assuming probabilities
        nonmember_preds = np.array(results["nonmember_preds"])  # Assuming probabilities

        # **Check for empty predictions**
        if member_preds.size == 0 or nonmember_preds.size == 0:
            print(f"⚠️ Class {cls} has empty predictions. Assigning NaN/0 values.")
            metrics[cls] = {
                "Accuracy": np.nan,
                "TPR": np.nan,
                "TNR": np.nan,
                "FPR": np.nan,
                "FNR": np.nan,
                "Advantage": np.nan,
                "AUC": np.nan
            }
            continue  # Move to the next class

        # Ensure they're probabilities (apply sigmoid if needed)
        if member_preds.max() > 1 or member_preds.min() < 0:  # Likely logits
            member_preds = torch.sigmoid(torch.tensor(member_preds)).numpy()
            nonmember_preds = torch.sigmoid(torch.tensor(nonmember_preds)).numpy()
        
        # Labels
        member_labels = np.ones(len(member_preds))  # Ground truth for members
        nonmember_labels = np.zeros(len(nonmember_preds))  # Ground truth for non-members
        
        # Predictions (Thresholded at 0.5)
        eps = 1e-10
        member_preds = np.clip(member_preds, eps, 1 - eps)
        nonmember_preds = np.clip(nonmember_preds, eps, 1 - eps)

        member_predictions = (member_preds > 0.5).astype(float)
        nonmember_predictions = (nonmember_preds < 0.5).astype(float)

        # Concatenate predictions and labels
        full_predictions = np.concatenate([member_predictions, nonmember_predictions])
        full_labels = np.concatenate([member_labels, nonmember_labels])

        # Compute confusion matrix components
        TP = np.sum((member_predictions == 1.0) & (member_labels == 1.0))
        FN = np.sum((member_predictions == 0.0) & (member_labels == 1.0))
        TN = np.sum((nonmember_predictions == 0.0) & (nonmember_labels == 0.0))
        FP = np.sum((nonmember_predictions == 1.0) & (nonmember_labels == 0.0))
        
        # Compute Metrics
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
        FPR = 1 - TNR
        FNR = 1 - TPR
        Adv = TPR - FPR  # Attack Advantage

        try:
            AUC = roc_auc_score(full_labels, np.concatenate([member_preds, nonmember_preds]))
        except ValueError:  # This happens if all predictions are the same class
            AUC = np.nan  

        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        
        # Store Metrics
        metrics[cls] = {
            "Accuracy": accuracy,
            "TPR": TPR,
            "TNR": TNR,
            "FPR": FPR,
            "FNR": FNR,
            "Advantage": Adv,
            "AUC": AUC
        }
    
    # **Compute average metrics while handling NaN values correctly**
    print("__________________________________________________")
    print("This is a debug")
    print(f"metrics : \n {metrics}")

    avg_metrics = {}
    valid_classes = [m for m in metrics.values() if not np.isnan(m["AUC"])]  # Ignore NaN classes for averaging
    if valid_classes:
        avg_metrics = {key: np.nanmean([m[key] for m in valid_classes]) for key in metrics[next(iter(metrics))]}
    else:
        avg_metrics = {key: np.nan for key in metrics[next(iter(metrics))]}  # If all are NaN, set NaN

    metrics["average"] = avg_metrics

    return metrics




def list_s3_files(bucket_name, prefix):
    """Lists all files under an S3 prefix."""
    s3_client = boto3.client("s3")
    
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    return [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".pt")]



def parse_checkpoint_name(checkpoint_name):
    """
    Parses the checkpoint name to extract client ID and iteration.
    
    Args:
        checkpoint_name (str): Checkpoint filename (e.g., "node_0_iteration_0.pt").

    Returns:
        dict: Parsed information with keys {"client_id": int or "orchestrator", "iteration": int}.
    """
    match = re.match(r"node_(\d+)_iteration_(\d+)\.pt", checkpoint_name)
    orchestrator_match = re.match(r"node_orchestrator_iteration_(\d+)\.pt", checkpoint_name)
    
    if match:
        return {"client_id": int(match.group(1)), "iteration": int(match.group(2))}
    elif orchestrator_match:
        return {"client_id": "orchestrator", "iteration": int(orchestrator_match.group(1))}
    else:
        return None


def run_attack_multiple_checkpoints(args, prefix, type_node="nodes"):
    s3_client = boto3.client("s3")
    prefix_models = f"{args.target_models_prefix}nodes/" if type_node == "nodes" else f"{args.target_models_prefix}orchestrator/"
    checkpoints = list_s3_files(args.target_models_bucket, prefix_models)  

    all_results = {}
    device = "cpu"
    
    local_attack_model_paths = download_attack_models_from_s3(args)
    attack_models = {cls: load_attack(local_path).to(device) for cls, local_path in local_attack_model_paths.items()}

    for checkpoint in checkpoints:
        local_checkpoint = os.path.basename(checkpoint)
        parsed_info = parse_checkpoint_name(local_checkpoint)

        if parsed_info is None:
            print(f"Skipping unrecognized checkpoint: {checkpoint}")
            continue

        client_id = parsed_info["client_id"]
        iteration = parsed_info["iteration"]

        print(f"Processing {checkpoint} (Client: {client_id}, Iteration: {iteration})...")
        s3_client.download_file(args.target_models_bucket, checkpoint, local_checkpoint)
        members, non_members = load_datasets(args, client_id)
        target_model = load_target(local_checkpoint)

        attack_metrics = run_attack_local(args, target_model, attack_models, members, non_members)

        if client_id not in all_results:
            all_results[client_id] = {}

        all_results[client_id][iteration] = attack_metrics
       

    all_iterations = [all_results[c][i] for c in all_results for i in all_results[c]]
    
    all_flat_results = []
    for result_dict in all_iterations:  
        all_flat_results.extend(result_dict.values())  

    avg_metrics = {}
    if all_flat_results:  
        for key in all_flat_results[0]:  
            values = [m[key] for m in all_flat_results if key in m and isinstance(m[key], (int, float, np.number))]
            if values:  
                avg_metrics[key] = np.mean(values)
            else:
                print(f"⚠ Skipping key '{key}' due to non-numeric values:", [m[key] for m in all_flat_results if key in m])

    all_results["average"] = avg_metrics

    #print("final debug of what will be saved")
    #print(all_results)

    # ✅ Ensure the directory exists
    output_dir = "/tmp/processing"  # Or another accessible directory
    os.makedirs(output_dir, exist_ok=True)

    # ✅ Use absolute path
    local_results_file = os.path.join(output_dir, f"{type_node}_attack_results.json")

    print(f"✅ Writing results to {local_results_file}")
    with open(local_results_file, "w") as f:
        json.dump(all_results, f, indent=4)

    output_s3_path = f"s3://{args.results_bucket}/{args.results_prefix}/{type_node}_attack_results.json"
    s3_client.upload_file(local_results_file, args.results_bucket, f"{args.results_prefix}/{type_node}_attack_results.json")

    print(f"✅ Results uploaded to {output_s3_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_bucket", type=str, required=True)
    parser.add_argument("--attack_model_prefix", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--dataset_bucket", type=str, required=True)
    parser.add_argument("--dataset_prefix", type=str, required=True)
    parser.add_argument("--target_models_prefix", type=str, required=True)
    parser.add_argument("--target_models_bucket", type=str, required=True)
    parser.add_argument("--results_bucket", type=str, required=True)
    parser.add_argument("--results_prefix", type=str, required=True)
    
    args = parser.parse_args()
    
    # Define the full paths for nodes and orchestrator checkpoints
    nodes_prefix = f"s3://{args.target_models_bucket}/{args.target_models_prefix}/nodes"
    orchestrator_prefix = f"s3://{args.target_models_bucket}/{args.target_models_prefix}/orchestrator"
    
    # Run attack separately for nodes and orchestrator checkpoints
    nodes_results = run_attack_multiple_checkpoints(args,nodes_prefix, type_node="nodes")
    
    orchestrator_results = run_attack_multiple_checkpoints(args, orchestrator_prefix, type_node="orchestrator")
    
    # Save results locally (SageMaker will upload them automatically)
    output_path = "/opt/ml/processing/output/attack_results.json"
    with open(output_path, "w") as f:
        json.dump({"nodes": nodes_results, "orchestrator": orchestrator_results}, f, indent=4)
    
    print(f"Saved attack results to {output_path}")
    print("Finished!")
    
    
