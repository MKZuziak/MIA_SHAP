# Installation Guide
1. Clone this git repository
```git clone https://github.com/MKZuziak/MIA_SHAP.git```
2. Navigate into the project's directory
```cd MIA_SHAP```
3. Install Poetry environment and dependencies.
```poetry install```
4. Run 'Poetry Shell' command
'''poetry shell'''

Note: The most convenient use is first entering into the VN (by invoking the poetry shell command), then launching particular scripts. An alternative way of launching scripts within the created environment
is to run ```poetry run python my_script.py``` command.

# Exemplary Simulation Script
This section provides a walkthrough an exemplary simulation script.
1. Define imports:

```
import os
from functools import partial

from torch import optim

from tests.test_props.datasets import return_mnist
from tests.test_props.nets import NeuralNetwork
from SHAP_MIA.model.federated_model import FederatedModel
from SHAP_MIA.node.federated_node import FederatedNode
from SHAP_MIA.simulation.simulation import Simulation
from SHAP_MIA.aggregators.fedopt_aggregator import Fedopt_Optimizer
from SHAP_MIA.files.archive import create_archive
```

All internal imports should load due to Poetry environment relative path managements.

2. Invoke create archive function

```
(metrics_savepath, 
nodes_models_savepath, 
orchestrator_model_savepath) = create_archive(os.getcwd(),
                                                archive_name='test_1_results')
```

Create archive function is a helper function that will create a structured archive in the gived directory
and will return three absolute paths: metrics_savepath, nodes_model_savepath and orchestrator_model_savepath.

3. Define Neural Network architecture

```
net_architecture = NeuralNetwork()
```

Creates a new object using standard PyTorch API.

4. Define (local) optimizer architecture.

```
optimizer_architecture = partial(optim.SGD, lr=0.001)
```

Defines local optimizer architecture with the help of partial from functools.

5. Define a (node's) model template.

```
model_tempate = FederatedModel(
    net=net_architecture,
    optimizer_template=optimizer_architecture,
    loader_batch_size=32
)
```

The following line creates a (general) template of the local model that will be copied each time a new node is
created. During one simulation run, the nodes templates are persistent. It means that all information saved on that node are carrying over within the realm of one simulation.

6. Create a node and optimizer template

```
node_template = FederatedNode()
fed_avg_aggregator = Fedopt_Optimizer()
```

7. Create a simulation instance

```
simulation_instace = Simulation(model_template=model_tempate,
    node_template=node_template)
```

Simulation object is a central entity that encapsulates all the states related to the simulation.

8. Load datasets

```
train, test = return_mnist()
```

Load datasets in the HuggingFace's Arrow format. In this case, we load only train and test MNIST dataset from
the HuggingFace Hub.

8. Attach datasets to particular clients

```
simulation_instace.attach_orchestrator_model(orchestrator_data=test)
simulation_instace.attach_node_model({
    3: [train, test],
    7: [train, test],
    11: [train, test],
    12: [train, test]
})
```

Attach datasets to the models. They should be loaded in the HuggingFace's Arrow format. In this case, we simply
replicate one dataset over all the clients.

9. Run simulation

```
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
```

Runs the simulation with pre-specified arguments.

10. Presenting it all together

```
import os
from functools import partial

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
     orchestrator_model_savepath) = create_archive(os.getcwd(),
                                                   archive_name='test_1_results')
    
    
    train, test = return_mnist()
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
    simulation_instace.attach_orchestrator_model(orchestrator_data=test)
    simulation_instace.attach_node_model({
        3: [train, test],
        7: [train, test],
        11: [train, test],
        12: [train, test]
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
```