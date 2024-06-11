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
        nodes: int,
        global_iterations: int,
        processing_batch: int
        ) -> None:
        self.epoch_shapley = {
            node_id: {
                global_iteration: 0 for global_iteration in global_iterations
                } for node_id in nodes
        }
        self.total_shapley = {
            node_id: 0 for node_id in nodes
        }
        self.processing_batch = processing_batch
    
    
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
            gradients.keys()
        )
        operation_counter = 0 # Operation counter for tracking and printing the progress
        number_of_operations = 2 ** (len(possible_coalitions.keys())) - 1
        recorded_values = {} # Maps every coalition to it's value, implemented to decrease the time complexity.
        
        if len(possible_coalitions.keys()) < self.processing_batch:
            self.processing_batch = len(possible_coalitions.keys())
        batches = chunker(
            seq = possible_coalitions,
            size = self.processing_batch
        )
        # Part I: compute value of particular coalitions in parallel
        for batch in batches:
            with Pool(self.processing_batch) as pool:
                results = [pool.apply_async(
                    calculate_coalition_value(
                        coalition,
                        copy.deepcopy(gradients),
                        copy.deepcopy(previous_weights),
                        model_template,
                        aggregator_template))
                           for coalition in batch]
                for result in results:
                    coalition_value = result.get()
                    recorded_values.update(coalition_value)
            operation_counter += len(batch)
            print(f"Completed {operation_counter} out of {number_of_operations} operations")
        print("Finished evaluating all of the coalitions. Commencing calculation of individual Shapley values.")
        for node in gradients.keys():
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
                shap += divisor * (coalition_with_client_score - coalition_without_client_score)
            self.partial_shapley[iteration][node] = shap / len(gradients.keys())
        
                
def calculate_coalition_value(
    coalition: list,
    gradients: OrderedDict,
    previous_weights: OrderedDict,
    model_template: FederatedModel,
    aggregator_template: Aggregator
    ) -> tuple[int, dict, float]:
        
        result = {}
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
        score = model_template.evaluate_model()[1]
        result[tuple(sorted(coalition))] = score
        return result


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