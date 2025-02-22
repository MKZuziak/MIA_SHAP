import pickle

import medmnist

from SHAP_MIA.dataset_generation.utils import *

# Loading the dataset
TissueMNIST = medmnist.TissueMNIST(split='train', size=28, download=True)
# Joining the val and train dataset together
TissueMNIST = convert_to_hg(TissueMNIST)
# Train-test (Orchestrator - nodes) split
TissueMNIST = TissueMNIST.train_test_split(test_size=0.2)
TissueMNIST_train = TissueMNIST['train']
TissueMNIST_test = TissueMNIST['test']

agents_cluster_belonging = {
            1: [0],
            2: [1],
            3: [2],
            4: [3],
            5: [4]
        } # Every seperate client has a its own separate 'cluster'

missing_classes = {
            1: [4, 5, 6, 7], # classes 0, 1, 2, 3 present only
            2: [3, 5, 6, 7], # classes 0, 1, 2, 4 present only
            3: [3, 4, 6, 7], # classes 0, 1, 2, 5 present only
            4: [3, 4, 5, 7], # classes 0, 1, 2, 6 present only
            5: [3, 4, 5, 6], # classes 0, 1, 2, 7 present only
        } # Classes 0, 1, 2 are shared 

TissueMNIST_train_nodes = missing_classes_clustered(
    dataset=TissueMNIST_train,
    shards=5,
    agents_cluster_belonging=agents_cluster_belonging,
    missing_classes_dict=missing_classes,
    seed=42,
    shuffle=True,
    local_test_size=0.3,
)

# Naming convention for a dataset saves
pointers_name = "TISSUEMNIST_5_dataset_pointers"
dataset_name = "TISSUEMNIST_5_dataset"
blueprint_name = "TISSUEMNIST_5_dataset_blueprint"
# Path to a directory
path = os.getcwd()
# Concatenating the test and train dataset into one
TissueMNIST_split = [TissueMNIST_test, TissueMNIST_train_nodes]
# Saving the dataset pointers
with open(os.path.join(path, pointers_name), "wb") as file:
    pickle.dump(TissueMNIST_split, file)
# Saving the original dataset
save_dataset(path=path, name=dataset_name, loaded_dataset = TissueMNIST_split)
# Saving the blueprint (distribution over labels) of the dataset
save_blueprint(TissueMNIST_split, path, blueprint_name)