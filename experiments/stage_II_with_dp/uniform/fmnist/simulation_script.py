import os
from functools import partial
import pickle

import timm

from torch import optim

from tests.test_props.datasets import return_mnist
from tests.test_props.nets import NeuralNetwork
from SHAP_MIA.model.federated_model import FederatedModel
from SHAP_MIA.node.federated_node import FederatedNode
from SHAP_MIA.simulation.simulation import Simulation
from SHAP_MIA.aggregators.fedopt_aggregator import Fedopt_Optimizer
from SHAP_MIA.files.archive import create_archive
from opacus.validators import ModuleValidator
from opacus import PrivacyEngine

DATASET_PATH = r'/Users/soumia.elmestari/Desktop/save/MIA_SHAP/experiments/datasets/uniform/fmnist/FMNIST_8_dataset_pointers'
NET_ARCHITECTURE = timm.create_model('resnet18', num_classes=10, pretrained=False, in_chans=1)
NUMBER_OF_CLIENTS = 8
ITERATIONS = 50
LOCAL_EPOCHS = 2
LOADER_BATCH_SIZE = 32
LEARNING_RATE = 0.001
ARCHIVE_PATH = os.getcwd()
ARCHIVE_NAME = 'withoutDP_uniform_fmnist'


def integration_test():
    (metrics_savepath, 
     nodes_models_savepath, 
     orchestrator_model_savepath) = create_archive(
         path=ARCHIVE_PATH,
         archive_name=ARCHIVE_NAME
     )
    
    with open(DATASET_PATH, 'rb') as file:
        data = pickle.load(file)
    orchestrators_data = data[0]
    nodes_data = data[1]
    net_architecture = NET_ARCHITECTURE
    optimizer_architecture = partial(optim.SGD, lr=LEARNING_RATE)
    ##########################
   

    privacy_engine = PrivacyEngine()
    net_architecture = ModuleValidator.fix(net_architecture)

    ##########################
    model_tempate = FederatedModel(
        net=net_architecture,
        optimizer_template=optimizer_architecture,
        loader_batch_size=LOADER_BATCH_SIZE
    )
    node_template = FederatedNode()
    fed_avg_aggregator = Fedopt_Optimizer()
    
    simulation_instace = Simulation(model_template=model_tempate,
                                    node_template=node_template)
    simulation_instace.attach_orchestrator_model(orchestrator_data=orchestrators_data)
    simulation_instace.attach_node_model({
            node: nodes_data[node] for node in range(NUMBER_OF_CLIENTS)
        })
    simulation_instace.training_protocol(
        iterations=ITERATIONS,
        sample_size=NUMBER_OF_CLIENTS,
        local_epochs=LOCAL_EPOCHS,
        aggrgator=fed_avg_aggregator,
        shapley_processing_batch=NUMBER_OF_CLIENTS,
        metrics_savepath=metrics_savepath,
        nodes_models_savepath=nodes_models_savepath,
        orchestrator_models_savepath=orchestrator_model_savepath
    )


if __name__ == "__main__":
    integration_test()