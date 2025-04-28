import timm
import torch
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
import numpy as np

from pathlib import Path
import re
import os 
import torch.serialization
import utils as UT
from torch.utils.data import DataLoader 
import torch.nn.functional as F
from attack_models import Attack_model

PATH = 'C:PATH_TO_FOLDER/models' # TO CHANGE ACCORDINGLY
datasets = 'C:/PATH_TO_FOLDER/MIA_SHAP/experiments/datasets'  # TO CHANGE ACCORDINGLY
base_folder = "C:/PATH_TO_FOLDER/MNIST_8_dataset" # TO CHANGE ACCORDINGLY

def load_checkpoints(paths):
    "returned loaded checkpoints"

    checkpoints = []
    for path in paths:
        instance = dict()
        arch = timm.create_model('resnet18', num_classes=10, pretrained=False, in_chans=1)

        checkpoint = torch.load(path,weights_only=False)
        arch.load_state_dict(checkpoint)
        file_name = re.split(r'[_\.\/\\]',path)
        type = file_name[-5]
        iteration = file_name[-2]
       
        num = file_name[-4]
        if file_name[-4] == 'orchestrator':
            type = 'orchestrator'
            num = None

                
        instance['type'] = type
        instance['number'] = num
        instance['iteration'] = iteration

        instance['model'] = arch

        checkpoints.append(instance)

    return checkpoints

def load_attack_models(paths):
    "returned loaded checkpoints"

    checkpoints = []
    for path in paths:
        instance = dict()
        arch = Attack_model(10)

        checkpoint = torch.load(path,weights_only=False)
        arch.load_state_dict(checkpoint)
        file_name = re.split(r'[_\.\/\\]',path)
        type = file_name[-1]

                
        instance['type'] = type
       

        instance['model'] = arch

        checkpoints.append(instance)

    return checkpoints

def list_files_in_directory(root_dir):
    file_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths




def evaluate_attack_model(attack_models,shadow_models,dataloaders,device ='cuda'):
    """
    Evaluate the attack models on the given datasets

    Parameters
    ----------
    attack_models : list
        List of attack models to evaluate
    shadow_models : list
        List of shadow models to use for evaluation
    dataloaders : list
        List of dataloaders to use for evaluation
    device : str, optional
        Device to use for evaluation (default is 'cuda')

    Returns
    -------
    metrics : dict
        Dictionary containing the metrics for each attack model
    avg_metrics : dict
        Dictionary containing the average metrics for all attack models
    """
    
    metrics={
        "accuracy":[],
        "precision":[],
        "recall":[],
        "f1":[],
        "AUC":[],
        "False_positive":[],
        "False_negative":[],
        "True_positive":[],
        "True_negative":[]
    }
    for i, (attack_model,data_loader) in enumerate(zip(attack_models,dataloaders)):
        attack_model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        attack_model = attack_model.to(device)
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                shadows_guess= []
                for shadow in shadow_models:
                    shadow = shadow.to(device)
                    shadow.eval()
                    guess = shadow(inputs)
                    classes = torch.softmax(guess,dim=0)
                    predicted_classes = torch.argmax(classes,dim=1)
                    # get the probs for only class i 
                    class_mask = (predicted_classes == i)
                    filtered_output = classes[class_mask]
                    
                    shadows_guess.append(filtered_output)

                for attack_input in shadows_guess:
                    outputs = attack_model(attack_input)

                    probs = torch.sigmoid(outputs)

                    preds = (probs > 0.5).int()

                    all_preds.append(preds.cpu().numpy)
                    all_labels.append(labels.cpu().numpy())
                    all_probs.append(probs.cpu().numpy)

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels,axis=0)
        all_probs = np.concatenate(all_probs,axis=0)

        acc = accuracy_score(all_labels,all_preds)
        precision = precision_score(all_labels,all_preds)
        recall = recall_score(all_labels,all_preds)
        f1 = f1_score(all_labels,all_preds)
        aoc = roc_auc_score(all_labels,all_probs)
        cm = confusion_matrix(all_labels,all_preds)
        TN,FP,FN,TP = cm.ravel()


        metrics["accuracy"].append(acc)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1"].append(f1)
        metrics["AUC"].append(aoc)
        metrics["True_negative"].append(TN)
        metrics["False_positive"].append(FP)
        metrics["False_negative"].append(FN)
        metrics["True_positive"].append(TP)


    avg_metrics = {key:np.mean(value) for key,value in metrics.items()}

    return metrics, avg_metrics


if __name__ == "__main__":
    attack_dir = "C:/PATH_TO_FOLDER/MIA_ATTACK/attack_models" # TO CHANGE ACCORDINGLY
    attck_files = list_files_in_directory(attack_dir)

    attack_models = load_attack_models(attck_files)

    files_models = list_files_in_directory(PATH)
    
    torch.serialization.add_safe_globals({
        'load_checkpoints' : load_checkpoints,
    })
    checkpoints = load_checkpoints(files_models)
    datasets = UT.load_client_ds(base_folder)

    dataloaders = []
    shadow_models = []
    
    for checkpoint in checkpoints:
        dataloader = datasets[checkpoint['number']]
        shadow_model = checkpoint['model']
        dataloaders.append(dataloader)
        shadow_models.append(shadow_model)  

    for attack_model in attack_models:
        attack_model.eval() 

        metrics, avg_metrics = evaluate_attack_model(attack_models, shadow_models, dataloaders, device='cuda')
        print(avg_metrics)
        print("-----------------------")


    

    