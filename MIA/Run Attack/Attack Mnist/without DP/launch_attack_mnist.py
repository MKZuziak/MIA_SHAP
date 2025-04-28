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
import random
# Install timm if not available
import subprocess
import sys
try:
    import timm
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip","-q", "install", "timm"])
    import timm  # Retry import after installation


from datasets import Dataset as HFDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,  classification_report, roc_auc_score, confusion_matrix
)
import torch.nn as nn
from collections import defaultdict
import re
import argparse

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

class Attack_model(nn.Module):
    def __init__(self,input_size):
        super(Attack_model,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size,64),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Sigmoid()
        )
    def forward(self,x):
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


############ a try for balanced 


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
    model = timm.create_model('resnet18', num_classes=10, pretrained=False, in_chans=1)
    model.load_state_dict(torch.load(checkpoint_path,map_location=torch.device("cpu")))
    model.eval()
    return model
     


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
        # Ensure the predictions exist in results
        member_preds = results.get("member_preds", [])
        nonmember_preds = results.get("nonmember_preds", [])
        
        # Convert to NumPy arrays safely
        member_preds = np.array(member_preds) if isinstance(member_preds, (list, np.ndarray, torch.Tensor)) else np.array([])
        nonmember_preds = np.array(nonmember_preds) if isinstance(nonmember_preds, (list, np.ndarray, torch.Tensor)) else np.array([])
        
        # Handle scalar case
        if member_preds.ndim == 0:
            member_preds = np.array([member_preds])
        if nonmember_preds.ndim == 0:
            nonmember_preds = np.array([nonmember_preds])
        
        # If both arrays are empty, assign NaN metrics and continue
        if member_preds.size == 0 or nonmember_preds.size == 0:
            print(f"⚠️ Skipping class {cls} due to no samples. Assigning NaN metrics.")
            metrics[cls] = {
                "Accuracy": np.nan,
                "TPR": np.nan,
                "TNR": np.nan,
                "FPR": np.nan,
                "FNR": np.nan,
                "Advantage": np.nan,
                "AUC": np.nan
            }
            continue
        
        # Ensure predictions are probabilities
        if member_preds.max() > 1 or member_preds.min() < 0:
            member_preds = torch.sigmoid(torch.tensor(member_preds)).numpy()
            nonmember_preds = torch.sigmoid(torch.tensor(nonmember_preds)).numpy()
        
        # Create labels for members and non-members
        member_labels = np.ones(len(member_preds))
        nonmember_labels = np.zeros(len(nonmember_preds))
        
        # Clip predictions
        eps = 1e-10
        member_preds = np.clip(member_preds, eps, 1 - eps)
        nonmember_preds = np.clip(nonmember_preds, eps, 1 - eps)

        # Threshold predictions
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
        
        # Calculate metrics safely
        TPR = TP / (TP + FN) if (TP + FN) > 0 else np.nan
        TNR = TN / (TN + FP) if (TN + FP) > 0 else np.nan
        FPR = 1 - TNR if TNR is not np.nan else np.nan
        FNR = 1 - TPR if TPR is not np.nan else np.nan
        Adv = TPR - FPR if TPR is not np.nan and FPR is not np.nan else np.nan
        
        # Compute AUC safely
        try:
            AUC = roc_auc_score(full_labels, np.concatenate([member_preds, nonmember_preds]))
        except ValueError:
            AUC = np.nan

        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else np.nan
        
        metrics[cls] = {
            "Accuracy": accuracy,
            "TPR": TPR,
            "TNR": TNR,
            "FPR": FPR,
            "FNR": FNR,
            "Advantage": Adv,
            "AUC": AUC
        }
    
    # Compute average metrics
    if metrics:
        keys = list(metrics[next(iter(metrics))].keys())
        avg_metrics = {key: np.nanmean([m[key] for m in metrics.values()]) for key in keys}
        metrics["average"] = avg_metrics
    else:
        metrics["average"] = {}

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


def run_attack_multiple_checkpoints_(args,prefix , type_node="nodes"):
    """
    Runs membership inference attack on multiple target model checkpoints.

    Args:
        target_model_prefix (str): S3 folder containing target model checkpoints.
        attack_bucket (str): S3 bucket storing attack models.
        attack_model_prefix (str): Prefix where attack models are stored.
        members_loader (DataLoader): Members dataset.
        nonmembers_loader (DataLoader): Non-members dataset.
        output_s3_prefix (str): S3 folder to store attack results.
    """
    s3_client = boto3.client("s3")
    #bucket_name, prefix = target_model_prefix.replace("s3://", "").split("/", 1)
    prefix_models = f"{args.target_models_prefix}nodes/" if type_node=="nodes" else f"{args.target_models_prefix}/orchestrator/"
    checkpoints = list_s3_files(args.target_models_bucket, prefix_models)  # List files in the target folder

    all_results = {}
    device = "cpu"
    # Generate attack model paths using the attack bucket
    #attack_model_paths = {i: f"s3://{args.attack_bucket}/{args.attack_model_prefix}/attack_class_{i}.pt" for i in range(10)}

    
    # Ensure attack models are downloaded before loading them
    local_attack_model_paths = download_attack_models_from_s3(args)
    
    # Load all attack models from local paths
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

        attack_metrics = run_attack_local(args,target_model, attack_models,  members, non_members)

        if client_id not in all_results:
            all_results[client_id] = {}

        all_results[client_id][iteration] = attack_metrics
        
    # Compute averaged metrics across all checkpoints
    all_iterations = [all_results[c][i] for c in all_results for i in all_results[c]]
    
    # Collect all metric dictionaries in a flat list
    all_flat_results = []
    for result_dict in all_iterations:  # Each item is a dictionary
        all_flat_results.extend(result_dict.values())  # Get inner dictionaries

    

    # Compute average metrics
    avg_metrics = {}
    if all_flat_results:  # Ensure there are results to process
        for key in all_flat_results[0]:  # Use first result's keys as reference
            values = [m[key] for m in all_flat_results if key in m and isinstance(m[key], (int, float, np.number))]
    
            if values:  # Compute mean only if we have valid numbers
                avg_metrics[key] = np.mean(values)
            else:
                print(f"⚠ Skipping key '{key}' due to non-numeric values:", [m[key] for m in all_flat_results if key in m])

    # Store averaged results
    all_results["average"] = avg_metrics

    print("final debug of what willl be saved")
    print(all_results)
    
     #Save results to S3
    local_results_file = f"{type_node}_attack_results.json"
    with open(local_results_file, "w") as f:
        json.dump(all_results, f, indent=4)

    output_s3_path = f"s3://{args.results_bucket}/{args.results_prefix}/{type_node}_attack_results.json"
    s3_client.upload_file(local_results_file, args.results_bucket, f"{args.results_prefix}/{type_node}_attack_results.json")
    print(f"Results uploaded to {output_s3_path}")


    return all_results

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
    
    
