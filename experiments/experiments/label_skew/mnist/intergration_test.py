import os
from functools import partial
import pickle

from torch import optim

from SHAP_MIA.model.federated_model import FederatedModel
from SHAP_MIA.node.federated_node import FederatedNode
from SHAP_MIA.simulation.simulation import Simulation
from SHAP_MIA.aggregators.fedopt_aggregator import Fedopt_Optimizer
from SHAP_MIA.files.archive import create_archive
from model import MNIST_Expanded_CNN

def simulation_run():
    (metrics_savepath, 
     nodes_models_savepath, 
     orchestrator_model_savepath) = create_archive(os.getcwd())
    root_dataset = r''
    with open(root_dataset, 'rb') as file:
        data = pickle.load(file)
    
    orchestrators_data = data[0]
    nodes_data = data[1]
    net_architecture = MNIST_Expanded_CNN()
    optimizer_architecture = partial(optim.SGD, lr=0.001)
    model_tempate = FederatedModel(
        net=net_architecture,
        optimizer_template=optimizer_architecture,
        loader_batch_size=32
    )
    node_template = FederatedNode()
    fed_avg_aggregator = Fedopt_Optimizer()
    
    simulation_instace = Simulation(
        model_template=model_tempate,
        node_template=node_template
        )
    simulation_instace.attach_orchestrator_model(orchestrator_data=orchestrators_data)
    simulation_instace.attach_node_model({
        nodes_id: nodes_data[nodes_id]
        for nodes_id in range(12)
    })
    simulation_instace.training_protocol(
        iterations=50,
        sample_size=12,
        local_epochs=2,
        aggrgator=fed_avg_aggregator,
        shapley_processing_batch=12,
        metrics_savepath=metrics_savepath,
        nodes_models_savepath=nodes_models_savepath,
        orchestrator_models_savepath=orchestrator_model_savepath
    )


if __name__ == "__main__":
    simulation_run()