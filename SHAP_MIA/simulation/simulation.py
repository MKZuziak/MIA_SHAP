from functools import partial
from collections import OrderedDict
from typing import Any
import copy
from multiprocessing import Pool
import os

import numpy as np

from SHAP_MIA.model.federated_model import FederatedModel
from SHAP_MIA.node.federated_node import FederatedNode
from SHAP_MIA.operations.orchestrations import train_nodes, sample_nodes
from SHAP_MIA.aggregators.aggregator import Aggregator
from SHAP_MIA.operations.evaluations import evaluate_model, automatic_node_evaluation
from SHAP_MIA.files.handlers import save_nested_dict_ascsv, save_partial_contribution, save_full_contribution
from SHAP_MIA.files.loggers import orchestrator_logger
from SHAP_MIA.shapley_calculation.shapley_calculation import Shapley_Calculation
from SHAP_MIA.loo_calculation.loo_calculation import LOO_Calculation
from SHAP_MIA.alpha_calculation.alpha_calculation import ALPHA_Calculation
from SHAP_MIA.utils.computations import average_of_weigts


# set_start_method set to 'spawn' to ensure compatibility across platforms.
from multiprocessing import set_start_method
set_start_method("spawn", force=True)
# Setting up the orchestrator logger
orchestrator_logger = orchestrator_logger()

class Simulation():
    """Simulation class representing a generic simulation type.
        
        Attributes
        ----------
        model_template : FederatedModel
            Initialized instance of a Federated Model class that is uploaded to every client.
        node_template : FederatedNode
            Initialized instance of a Federated Node class that is used to simulate nodes.
        data : dict
            Local data used for the training in a dictionary format, mapping each client to its respective dataset.
        """
    
    
    def __init__(self, 
                 model_template: FederatedModel,
                 node_template: FederatedNode,
                 seed: int = 42,
                 **kwargs
                 ) -> None:
        """Creating simulation instant requires providing an already created instance of model template
        and node template. Those instances then will be copied n-times to create n different nodes, each
        with a different dataset. Additionally, a data for local nodes should be passed in form of a dictionary,
        maping dataset to each respective client.

        Parameters
        ----------
        model_template : FederatedModel
            Initialized instance of a Federated Model class that will be uploaded to every client.
        node_template : FederatedNode
            Initialized instance of a Federated Node class that will be used to simulate nodes.
        seed : int,
            Seed for the simulation, default to 42
        **kwargs : dict, optional
            Extra arguments to enable selected features of the Orchestrator.
            passing full_debug to **kwargs, allow to enter a full debug mode.

        Returns
        -------
        None
        """
        self.model_template = model_template
        self.node_template = node_template
        self.network = {}
        self.orchestrator_model = None
        self.generator = np.random.default_rng(seed=seed)
    
    
    def attach_orchestrator_model(self,
                                  orchestrator_data: Any):
        """Attaches model of the orchestrator that is saved as an instance attribute.
        
        Parameters
        ----------
        orchestrator_data: Any 
            Orchestrator data that should be attached to the orchestrator model.
        
        Returns
        -------
        None
        """
        self.orchestrator_model = copy.deepcopy(self.model_template)
        self.orchestrator_model.attach_dataset_id(local_dataset=[orchestrator_data],
                                                  node_name='orchestrator',
                                                  only_test=True)
    
    
    def attach_node_model(
        self,
        nodes_data: dict,
        dp_settings: dict
        ):
        """Attaches models of the nodes to the simulation instance.
        
        Parameters
        ----------
        orchestrator_data: Any
            Orchestrator data that should be attached to nodes models.
        Returns
        -------
        None
        """
        for (node_id, data), (node_id_dp, dp_setting) in zip(nodes_data.items(), dp_settings.items()):
            assert node_id == node_id_dp
            self.network[node_id] = copy.deepcopy(self.node_template)
            self.network[node_id].connect_data_id(
                node_id = node_id,
                model = copy.deepcopy(self.model_template),
                data=data,
                dp=dp_setting['DP'],
                privacy_engine=dp_setting['Privacy_Engine']
                )
    

    def train_epoch(self,
                    sampled_nodes: dict[int: FederatedNode],
                    iteration: int,
                    local_epochs: int, 
                    mode: str = 'weights',
                    save_model: bool = False,
                    save_path: str = None) -> tuple[dict[int, int, float, float, list, list], dict[int, OrderedDict]]:
        """Performs one training round of a federated learning. Returns training
        results upon completion.
        
        Parameters
        ----------
        samples_nodes: dict[int: FederatedNode]
            Dictionary containing sampled Federated Nodes
        iteration: int
            Current global iteration of the training process
        local_epochs: int
            Number of local epochs for which the local model should
            be trained.
        mode: str (default to 'weights')
            Mode = 'weights': Node will return model's weights.
            Mode = 'gradients': Node will return model's gradients.
        save_model: bool (default to False)
            Boolean flag to save a model.
        save_path: str (defualt to None)
            Path object used to save a model

        Returns
        -------
        tuple[dict[int, int, float, float, list, list], dict[int, OrderedDict]]
        """
        training_results = {}
        weights = {}
        with Pool(len(sampled_nodes)) as pool:
            results = [pool.apply_async(train_nodes, (node, iteration, local_epochs, mode, save_model, save_path)) for node in list(sampled_nodes.values())]
            for result in results:
                node_id, model_weights, loss_list, accuracy_list = result.get()
                weights[node_id] = model_weights
                training_results[node_id] = {
                    "iteration": iteration,
                    "node_id": node_id,
                    "loss": loss_list[-1], 
                    "accuracy": accuracy_list[-1],
                    "full_loss": loss_list,
                    "full_accuracy": accuracy_list}
        return (training_results, weights)

        
    def training_protocol(self,
                          iterations: int,
                          sample_size: int,
                          local_epochs: int,
                          aggrgator: Aggregator,
                          shapley_processing_batch: int,
                          metrics_savepath: str,
                          nodes_models_savepath: str,
                          orchestrator_models_savepath: str) -> None:
        """Performs a full federated training according to the initialized
        settings. The train_protocol of the generic_orchestrator.Orchestrator
        follows a classic FedAvg algorithm - it averages the local weights 
        and aggregates them taking a weighted average.
        SOURCE: Communication-Efficient Learning of
        Deep Networks from Decentralized Data, H.B. McMahan et al.

        Parameters
        ----------
        iterations: int
            Number of (global) iterations // epochs to train the models for.
        sample_size: int
            Size of the sample
        local_epochs: int
            Number of local epochs for which the local model should
            be trained.
        aggregator: Aggregator
            Instance of the Aggregator object that will be used to aggregate the result each round
        shapley_processing_batch: int
            How many coalitions we want to calculate simultaneously. Higer number of coalitions speed
            up the process significantly, but are more demanding in terms of GPU VRAM.
        metrics_savepath: str
            Path for saving the metrics
        nodes_models_savepath: str
            Path for saving the models in the .pt format.
        orchestrator_models_savepath: str
            Path for saving the orchestrator models.
            
        Returns
        -------
        int
            Returns 0 on the successful completion of the training.
        """
        shapley_manager = Shapley_Calculation(
            global_iterations = iterations,
            processing_batch=shapley_processing_batch
        )
        loo_manager = LOO_Calculation(
            global_iterations = iterations
        )
        alpha_manager = ALPHA_Calculation(
            global_iterations = iterations,
            amplification = 5
        )
        
        for iteration in range(iterations):
            orchestrator_logger.info(f"Iteration {iteration}")
            
            # # Updating weights for every node
            # for node in self.network.values():
            #     node.update_weights(copy.deepcopy(self.central_model.get_weights()))
            
            # Sampling nodes
            sampled_nodes = sample_nodes(
                nodes = self.network,
                sample_size = sample_size,
                generator = self.generator
            )
            
            # Training nodes
            training_results, gradients = self.train_epoch(
                sampled_nodes=sampled_nodes,
                iteration=iteration,
                local_epochs=local_epochs,
                mode='gradients',
                save_model=True,
                save_path=nodes_models_savepath
            )
            
            shapley_manager.calculate_iteration_shapley(
                iteration = iteration,
                gradients = gradients,
                previous_weights = self.orchestrator_model.get_weights(),
                model_template = copy.deepcopy(self.orchestrator_model),
                aggregator_template = aggrgator
            )
            loo_manager.calculate_iteration_loo(
                iteration = iteration,
                gradients = gradients,
                previous_weights = self.orchestrator_model.get_weights(),
                model_template = copy.deepcopy(self.orchestrator_model),
                aggregator_template = aggrgator
            )
            alpha_manager.calculate_iteration_alpha(
                iteration = iteration,
                gradients = gradients,
                previous_weights = self.orchestrator_model.get_weights(),
                model_template = copy.deepcopy(self.orchestrator_model),
                aggregator_template = aggrgator
            )
            
            # Preserving metrics of the training
            save_nested_dict_ascsv(
                data=training_results,
                save_path=os.path.join(metrics_savepath, 'training_metrics.csv'))
            
            # Testing nodes on the local dataset before the model update (only sampled nodes).
            automatic_node_evaluation(
                iteration=iteration,
                nodes=sampled_nodes,
                save_path=os.path.join(metrics_savepath, "before_update_metrics.csv"))
            
            avg_gradients = average_of_weigts(gradients)
            # Updating weights
            new_weights = aggrgator.optimize_weights(
                weights=self.orchestrator_model.get_weights(),
                gradients = avg_gradients,
                learning_rate = 1.0,
                )
            
            # Updating weights for each node in the network
            for node in self.network.values():
                node.update_weights(new_weights)
            # Updating the weights for the central model
            self.orchestrator_model.update_weights(new_weights)
            
            # Preserving the orchestrator's model
            self.orchestrator_model.store_model_on_disk(iteration=iteration,
                                                        path=orchestrator_models_savepath)
            
            # Evaluating the new set of weights on local datasets.
            automatic_node_evaluation(
                iteration=iteration,
                nodes=self.network,
                save_path=os.path.join(metrics_savepath, "after_update_metrics.csv"))
            # Evaluating the new set of weights on orchestrator's dataset.
            evaluate_model(
                iteration=iteration,
                model=self.orchestrator_model,
                save_path=os.path.join(metrics_savepath, "orchestrator_metrics.csv"))
        shapley_manager.calculate_final_shapley(
            all_nodes_ids=self.network.keys(),
            total_iteration_no=iterations
        )
        loo_manager.calculate_final_loo(
            all_nodes_ids=self.network.keys(),
            total_iteration_no=iterations
        )
        alpha_manager.calculate_final_alpha(
            all_nodes_ids=self.network.keys(),
            total_iteration_no=iterations
        )
        save_partial_contribution(
            data=shapley_manager.epoch_shapley,
            save_path=os.path.join(metrics_savepath, 'partial_shapley.csv')
        )
        save_partial_contribution(
            data=loo_manager.epoch_loo,
            save_path=os.path.join(metrics_savepath, 'partial_loo.csv')
        )
        save_partial_contribution(
            data=alpha_manager.epoch_alpha,
            save_path=os.path.join(metrics_savepath, 'partial_alpha.csv')
        )
        save_full_contribution(
            data=shapley_manager.total_shapley,
            save_path=os.path.join(metrics_savepath, 'full_shapley.csv')
        )
        save_full_contribution(
            data=loo_manager.total_loo,
            save_path=os.path.join(metrics_savepath, 'full_loo.csv')
        )
        save_full_contribution(
            data=alpha_manager.total_alpha,
            save_path=os.path.join(metrics_savepath, 'full_alpha.csv')
        )
        orchestrator_logger.critical("Training complete")
        # return 0
                        