import torch
import torchvision
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,random_split,Subset
import  torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 
import timm
import os 
import random 
import numpy as np 
from torch.utils.data import TensorDataset
from models import Attack_model, ShadowModelEmnist
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
## shadows are trained on EMNIST 


NUM_SHADOWS= 5
SUBSET_SIZE = 0 
SHADOW_BATCH_SIZE= 64


def prepare_dataset():
    """
    LOAD THE DATASET AND SPLIT INTO SUBSETS FOR EACH SHADOW MODEL
    """
    #Define transformations that are similar to Mnist 
    transform = transforms.Compose([
        transforms.ToTensor()
    ])


    # load the shadow dataset , Choice is EMNIST
    emnist_dataset = datasets.EMNIST(
        root="./data",
        split="digits",
        train=True,
        download=True,
        transform=transform 
    )
    emnist_dataset_test  = datasets.EMNIST(
        root="./data",
        split="digits",
        train=False,
        download=True,
        transform=transform 
    )
   
    SUBSET_SIZE = len(emnist_dataset) // NUM_SHADOWS
    subsets = random_split(emnist_dataset,[SUBSET_SIZE]* (NUM_SHADOWS))
    return subsets,emnist_dataset_test,SUBSET_SIZE

def train_model(model,dataloader,epochs=5,device='cpu'):
    """
    RESTURNS A TRAINED SHADOW MODEL 
    """
    # we use the same models archtecture used to train the target model 
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs,labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs,labels)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            running_loss +=loss.item()

        print(f"Epoch {epoch +1}/{epochs}, Loss:{running_loss/len(dataloader)}")

    return model

def evaluate_model(model,eval_loader, device='cuda'):
    model.eval() 
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    total_val =0.0
    correct_val =0
    model = model.to('cuda')
    
    with torch.no_grad():
        for data, target in eval_loader:
            data,target = data.to('cuda'),target.to('cuda')
            output = model(data)
            loss = criterion(output,target)
            running_loss += loss.item()
            _,predicted = torch.max(output,1)

            total_val += target.size(0) 

            correct_val += (predicted == target).sum().item()
    
    val_loss = running_loss / len(eval_loader)
    val_acc = 100 * correct_val /total_val


    print(f'Validation Loss:{val_loss:.4f}, val acc = {val_acc:.4f}')

def train_shadow_models():
    """
    RESTURNS All TRAINED SHADOW MODEL  with their datasets 
    """
    device = torch.device("cuda" if torch.cuda.is_available()else "cpu")
    print(device)
    shadow_models = []

   

    subsets,test_subset,subset_size =  prepare_dataset()

 

    eval_loader = DataLoader(test_subset ,batch_size=SHADOW_BATCH_SIZE,shuffle=True )

    for i, subset in enumerate(subsets):
        print(f"Training shadow model {i + 1 }/{NUM_SHADOWS} ...")
        dataloader = DataLoader(subset,batch_size=SHADOW_BATCH_SIZE,shuffle=True )
        model = ShadowModelEmnist()#timm.create_model('resnet18', num_classes=10, pretrained=False, in_chans=1)
        trained_model = train_model(model,dataloader,epochs= 10,device=device)
        shadow_models.append(trained_model)
        evaluate_model(trained_model,eval_loader)

    os.makedirs('./shadow_models',exist_ok=True)
    for i, model in enumerate(shadow_models):
        torch.save(model.state_dict(),f"./shadow_models/shadow_model_{i}.pth")
        print(f"shadow model {i+1} has been saved.")

    return shadow_models,subsets


def generate_attack_data(shadow_models, datasets,num_classes):
    """
    GENERATE ATTACK DATASETS FOR EACH CLASS
    RETURNS:
        Attack_datasets : List of datasets for each class ( binary attack datasets)
    """
    attack_datasets = {cls:{"members":[],"non_members":[]} for cls in range(num_classes)}


    for cls in range(num_classes):
        

        for model,dataset in zip(shadow_models,datasets):
            dataloader = DataLoader(dataset,batch_size=SHADOW_BATCH_SIZE,shuffle=False)
            model.eval()

            for inputs,labels in dataloader:
                with torch.no_grad():
                    inputs = inputs.to('cuda')
                    outputs = model(inputs)
                    probs = torch.softmax(outputs,dim=0).cpu()

                    mask = labels == cls

                    mask_exclude = labels != cls
                    #print(probs[mask].shape)
                    kept = probs[mask]
                    notk = probs[mask_exclude]
                    if kept.shape[0] >0 :

                        confidence_vect = torch.chunk(kept, kept.shape[0],dim=0) #probs[mask].split(1,dim=0)
                         # members and non members 
                        for el in confidence_vect:
                        
                            attack_datasets[cls]['members'].append(el)
                    

                    if notk.shape[0] >0 :
                        confidence_vect_excluse_cls =  torch.chunk(notk, notk.shape[0],dim=0)#list(probs[mask_exclude].split(1,dim=0))
                        for e in confidence_vect_excluse_cls:

                            attack_datasets[cls]["non_members"].append(e)
                    
                
                    
                   
       
    return attack_datasets


def prepare_binary_attack_datasets(attack_datasets):
    """
    Convert attack datasets into TensorDatasets for training the attack models
    """
    binary_datasets = []

   
    for cls in attack_datasets:
        members = np.array(attack_datasets[cls]["members"])
        non_members = np.array(attack_datasets[cls]["non_members"])
        data = np.row_stack((members,non_members))
        labels = np.hstack((np.ones(len(members)),np.zeros(len(non_members))))

        binary_datasets.append(TensorDataset(torch.tensor(data,dtype=torch.float32,),
                                             torch.tensor(labels,dtype=torch.float32)))
        

    return binary_datasets

def train_attack_models(binary_datasets,input_size,num_epochs=4,batch_size=64):
    attack_models =[]

    for cls, dataset in enumerate(binary_datasets):
        print(f"Training attack model for class {cls} ...")
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
        model = Attack_model(input_size=input_size)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(),lr=0.001)

        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0 
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                #print(inputs.shape)

                outputs = model(inputs).squeeze()
                loss = criterion(outputs,labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print(f"Epoch {epoch +1 }, Loss: {running_loss/len(dataloader)}")
        attack_models.append(model)
    os.makedirs('./attack_models',exist_ok=True)
    for i, model in enumerate(attack_models):
        torch.save(model.state_dict(),f"./attack_models/attack_model_{i}.pth")
        print(f"attack model {i+1} has been saved.")


   
    return attack_models

if __name__ == "__main__":
    # start with shadows 
    shadow_models,subsets = train_shadow_models()
    # Attack models 
    attack_datasets= generate_attack_data(shadow_models, subsets,num_classes=10)
    binary_datasets = prepare_binary_attack_datasets(attack_datasets)
   


    # test dataset 
    

    attack_models = train_attack_models(binary_datasets,input_size=10,num_epochs=20)
    















