import math
import copy
from multiprocessing import Pool
from collections import OrderedDict
from _collections_abc import Generator

from SHAP_MIA.model.federated_model import FederatedModel
from SHAP_MIA.aggregators.aggregator import Aggregator
from SHAP_MIA.shapley_calculation.utils import form_superset, select_subsets, select_gradients
from SHAP_MIA.utils.computations import average_of_weigts


class Shapley_Calculation:
    def __init__(
        self,
        global_iterations: int,
        processing_batch: int
        ) -> None:
        self.epoch_shapley = {
            global_iteration: {} for global_iteration in range(global_iterations)
        }
        self.total_shapley = {}
        self.processing_batch = processing_batch
        self.metrics = [f'accuracy_per_{class_id}' for class_id in range(10)]
        self.metrics.extend(['test_loss', 'accuracy', 'f1score', 'precision', 'recall'])
    
    
    def calculate_iteration_shapley(
        self,
        iteration: int,
        gradients: OrderedDict,
        previous_weights: OrderedDict,
        model_template = FederatedModel,
        aggregator_template = Aggregator
    ):
        print(f"Calculating Shapley Score for iteration {iteration}")
        possible_coalitions = form_superset(
            gradients.keys(),
            return_dict=False
        )
        operation_counter = 0 # Operation counter for tracking and printing the progress
        number_of_operations = len(possible_coalitions)
        recorded_values = {} # Maps every coalition to it's value, implemented to decrease the time complexity.
        
        if len(possible_coalitions) < self.processing_batch:
            self.processing_batch = len(possible_coalitions)
        batches = chunker(
            seq = possible_coalitions,
            size = self.processing_batch
        )
        # Part I: compute value of particular coalitions in parallel
        for batch in batches:
            with Pool(self.processing_batch) as pool:
                results = [pool.apply_async(
                calculate_coalition_value,
                (coalition,
                 copy.deepcopy(gradients),
                 copy.deepcopy(previous_weights),
                 copy.deepcopy(model_template),
                 copy.deepcopy(aggregator_template)
                 )) for coalition in batch]
                
                for result in results:
                    coalition_value = result.get()
                    recorded_values.update(coalition_value)
            operation_counter += len(batch)
            print(f"Completed {operation_counter} out of {number_of_operations} operations")
        print("Finished evaluating all of the coalitions. Commencing calculation of individual Shapley values.")
        for node in gradients.keys():
            self.epoch_shapley[iteration][node] = {}
            for metric in self.metrics:
                shap = 0.0
                coalitions_of_interest = select_subsets(
                    coalitions = possible_coalitions,
                    searched_node = node
                )
                for coalition in coalitions_of_interest:
                    coalition_without_client = tuple(sorted(coalition))
                    coalition_with_client = tuple(sorted(tuple((coalition)) + (node,))) #TODO: Make a more elegant solution...
                    coalition_without_client_score = recorded_values[coalition_without_client]
                    coalition_with_client_score = recorded_values[coalition_with_client]
                    possible_combinations = math.comb((len(gradients.keys()) - 1), len(coalition_without_client))
                    divisor = 1 / possible_combinations
                    shap += divisor * (coalition_with_client_score[metric] - coalition_without_client_score[metric])
                self.epoch_shapley[iteration][node][metric] = shap / len(gradients.keys())
    
    
    def calculate_final_shapley(
        self,
        all_nodes_ids: list,
        total_iteration_no: int
    ):
        for node in all_nodes_ids:
            self.total_shapley[node] = {metric:0 for metric in self.metrics}
            for partial_results in self.epoch_shapley.values():
                for metric in self.metrics:
                    self.total_shapley[node][metric] += partial_results[node][metric]
            for metric in self.metrics:
                self.total_shapley[node][metric] = self.total_shapley[node][metric] / total_iteration_no
            
            
def calculate_coalition_value(
    coalition: list,
    gradients: OrderedDict,
    previous_weights: OrderedDict,
    model_template: FederatedModel,
    aggregator_template: Aggregator
    ) -> tuple[int, dict, float]:
        
        calc = {}
        coalitions_gradients = select_gradients(
            gradients = gradients,
            query = coalition
        )
        avg_gradients = average_of_weigts(coalitions_gradients)
        # Updating weights
        new_weights = aggregator_template.optimize_weights(
            weights = previous_weights,
            gradients = avg_gradients,
            learning_rate = 1.0,
            )
        
        model_template.update_weights(new_weights)
        eval_results = model_template.evaluate_model()
        results = {}
        results['test_loss'] = eval_results['test_loss']
        results['accuracy'] = eval_results['accuracy']
        results['f1score'] = eval_results['f1score']
        results['precision'] = eval_results['precision']
        results['recall'] = eval_results['recall']
        for class_id in range(10):
            results[f'accuracy_per_{class_id}'] = eval_results[f'accuracy_per_{class_id}']
        calc[tuple(sorted(coalition))] = results
        return calc


def chunker(
    seq: iter, 
    size: int
    ) -> Generator:
    """Helper function for splitting an iterable into a number
    of chunks each of size n. If the iterable can not be splitted
    into an equal number of chunks of size n, chunker will return the
    left-overs as the last chunk.
        
    Parameters
    ----------
    sqe: iter
        An iterable object that needs to be splitted into n chunks.
    size: int
        A size of individual chunk
    
    Returns
    -------
    Generator
        A generator object that can be iterated to obtain chunks of the original iterable.
    """
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))