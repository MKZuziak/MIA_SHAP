import math
import copy
from multiprocessing import Pool
from collections import OrderedDict

from SHAP_MIA.model.federated_model import FederatedModel
from SHAP_MIA.aggregators.aggregator import Aggregator
from SHAP_MIA.loo_calculation.utils import form_coalitions, select_gradients, select_subsets
from SHAP_MIA.utils.computations import average_of_weigts

class LOO_Calculation:
    def __init__(
        self,
        global_iterations: int
        ) -> None:
        self.epoch_loo = {
            global_iteration: {} for global_iteration in range(global_iterations)
        }
        self.total_loo = {}
        self.metrics = [f'accuracy_per_{class_id}' for class_id in range(10)]
        self.metrics.extend(['test_loss', 'accuracy', 'f1score', 'precision', 'recall'])
    
    
    def calculate_iteration_loo(
        self,
        iteration: int,
        gradients: OrderedDict,
        previous_weights: OrderedDict,
        model_template = FederatedModel,
        aggregator_template = Aggregator
    ):
        print(f"Calculating Leave-one-out Score for iteration {iteration}")
        possible_coalitions = form_coalitions(
            elements=gradients.keys(),
            return_dict=True
            ) # Leave-one-out coalitions
        grand_coalition = [node for node in gradients.keys()]
        recorded_values = {} # Maps every coalition to it's value, implemented to decrease the time complexity.
        # Part I: compute value of particular coalitions in parallel
        # Leave-one-out coalitions
        with Pool(len(possible_coalitions)) as pool:
            results = [pool.apply_async(
                calculate_coalition_value,
                (coalition,
                 copy.deepcopy(gradients),
                 copy.deepcopy(previous_weights),
                 copy.deepcopy(model_template),
                 copy.deepcopy(aggregator_template)
                 )) for coalition in possible_coalitions]
            for result in results:
                coalition_value = result.get()
                recorded_values.update(coalition_value)
        # Grand coalition
        recorded_values.update(
            calculate_coalition_value(
                coalition=grand_coalition,
                gradients=copy.deepcopy(gradients),
                previous_weights=copy.deepcopy(previous_weights),
                model_template=copy.deepcopy(model_template),
                aggregator_template=copy.deepcopy(aggregator_template))
        )
        print("Finished evaluating all of the coalitions. Commencing calculation of individual LOO values.")        
        # Part II: Computing LOO values using coalitions' values.
        for node in gradients.keys():
            self.epoch_loo[iteration][node] = {}
            for metric in self.metrics:
                coalition_with_client = tuple(sorted(grand_coalition))
                coalition_without_client = copy.deepcopy(grand_coalition)
                coalition_without_client.remove(node)
                coalition_without_client = tuple(sorted(coalition_without_client))
                coalition_with_client_score = recorded_values[coalition_with_client]
                coalition_without_client_score = recorded_values[coalition_without_client]
                self.epoch_loo[iteration][node][metric] = coalition_with_client_score[metric] - coalition_without_client_score[metric]
    
    
    def calculate_final_loo(
    self,
    all_nodes_ids: list,
    total_iteration_no: int
    ):
        for node in all_nodes_ids:
            self.total_loo[node] = {metric:0 for metric in self.metrics}
            for partial_results in self.epoch_loo.values():
                for metric in self.metrics:
                    self.total_loo[node][metric] += partial_results[node][metric]
            for metric in self.metrics:
                self.total_loo[node][metric] = self.total_loo[node][metric] / total_iteration_no


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