import os
from functools import partial
import pickle

from torch import optim

from tests.test_props.datasets import return_mnist
from tests.test_props.nets import NeuralNetwork
from SHAP_MIA.model.federated_model import FederatedModel
from SHAP_MIA.node.federated_node import FederatedNode
from SHAP_MIA.simulation.simulation import Simulation
from SHAP_MIA.aggregators.fedopt_aggregator import Fedopt_Optimizer
from SHAP_MIA.files.archive import create_archive

def integration_test():
    (metrics_savepath, 
     nodes_models_savepath, 
     orchestrator_model_savepath) = create_archive(os.getcwd())
    
    
    with open(r'/home/maciejzuziak/raid/MIA_SHAP/tests/integration_tests/test_3/custom_dataset/MNIST_4_dataset_pointers', 'rb') as file:
        data = pickle.load(file)
    orchestrators_data = data[0]
    nodes_data = data[1]
    net_architecture = NeuralNetwork()
    optimizer_architecture = partial(optim.SGD, lr=0.001)
    model_tempate = FederatedModel(
        net=net_architecture,
        optimizer_template=optimizer_architecture,
        loader_batch_size=32
    )
    node_template = FederatedNode()
    fed_avg_aggregator = Fedopt_Optimizer()
    
    simulation_instace = Simulation(model_template=model_tempate,
                                    node_template=node_template)
    simulation_instace.attach_orchestrator_model(orchestrator_data=orchestrators_data)
    simulation_instace.attach_node_model({
        2: nodes_data[0],
        9: nodes_data[1],
        14: nodes_data[2],
        18: nodes_data[3]
    })
    simulation_instace.training_protocol(
        iterations=15,
        sample_size=4,
        local_epochs=2,
        aggrgator=fed_avg_aggregator,
        shapley_processing_batch=8,
        metrics_savepath=metrics_savepath,
        nodes_models_savepath=nodes_models_savepath,
        orchestrator_models_savepath=orchestrator_model_savepath
    )


if __name__ == "__main__":
    integration_test()