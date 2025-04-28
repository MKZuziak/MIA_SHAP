import os 
from datasets import Dataset 
from torch.utils.data import Dataset as Torch_ds
import torch 
from torchvision import transforms



class ArrowToTorchDataset(Torch_ds):
    """ Convert a Hugging face dataset into a Pytorch Dataset"""
    def __init__(self,arrow_ds):
        self.arrow_ds = arrow_ds
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.arrow_ds)
    
    def __getitem__(self,idx):
        item = self.arrow_ds[idx]
        img = self.transform(item["image"])
        label = torch.tensor(item["label"])

        return img,label
    
def load_client_ds(base_folder):
    """
    Loads datasets from a directory structure (similar to the one we have )
    Args:
        base_folder (str) : path to the root folder
    Returns:
        dict : a dictionary containing :
                dataset : Pytorch dataset 
                client : client id 
                set_type : "train" or "test"
    """
    datasets ={}

    for subfolder in os.listdir(base_folder):
        if subfolder.startswith("client_"):
            subfolder_path = os.path.join(base_folder,subfolder)
            if os.path.isdir(subfolder_path):
                parts = subfolder.split("_")
                client_id = parts[1]
                set_type = "train" if "train" in subfolder else "test"
            
                for file_name in os.listdir(subfolder_path):
                    if file_name.endswith(".arrow"):
                        arrow_path = os.path.join(subfolder_path,file_name)

                        hf_ds = Dataset.from_file(arrow_path)
                        torch_ds = ArrowToTorchDataset(hf_ds)

                        key = f"client_{client_id}_{set_type}"
                        datasets[key] = {
                            "dataset": torch_ds,
                            "Client" : client_id,
                            "set_type" : set_type 
                            }
    return datasets


def extract_dict_element_by_attr(data, conds):
    """
    Extracts an element from a dictionary or a list of dictionaries based the value of specific attributes
    Args:
        data (list or dict): the input dict
        attributes (dict): {key:value}
    OUTPUT:
        dict or None based on whether the matching was found
        
    """
 
    matches = []
    def match(record,conds):
        return all(record.get(k) == v for k,v in conds.items())
    

    if isinstance(data,dict) and all(isinstance(v,dict) for v in data.values()):
        if match(data,conds):
            matches.append(data)
        else:
            for sub_key,sub_value in data.items():
                if  match(sub_value,conds):
                    matches.append(sub_value)
    return matches

if __name__ == "__main__":
    base_folder = "C:FOLDER_PATH/mnist/MNIST_8_dataset" # TO CHANGE ACCORDINGLY
    datasets = load_client_ds(base_folder)
    print(datasets)
    
