import csv
import os
import copy


def save_dict_ascsv(
    data: dict,
    save_path: str):
    """Attaches models of the nodes to the simulation instance.
        
    Parameters
    ----------
    data: dict
        Dictionary containing the data. Data in the dictionary will be transformed
        and saved under fieldnames formed by keys.
    save_path: str
        A full save path for preserving the dictionary. Must end with *.csv file extension.
    Returns
    -------
    None
    """
    with open(save_path, "a", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=next(iter(data.values())).keys())
        # Checks if the file already exists, otherwise writer header
        if os.path.getsize(save_path) == 0:
            writer.writeheader()
        for row in data.values():
            writer.writerow(row)


def save_nested_dict_ascsv(
    data: dict[dict],
    save_path: str):
    """Attaches models of the nodes to the simulation instance.
        
    Parameters
    ----------
    data: dict[dict]
        Nested dict containing the data. Note that the external key will not be saved, only the 
        data in the internal dictionary will be transformed to fieldnames and saved.
    save_path: str
        A full save path for preserving the dictionary. Must end with *.csv file extension.
    Returns
    -------
    None
    """
    with open(save_path, "a", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=next(iter(data.values())).keys())
        # Checks if the file already exists, otherwise writer header
        if os.path.getsize(save_path) == 0:
            writer.writeheader()
        for row in data.values():
            writer.writerow(row)


def save_partial_contribution(
    data: dict[dict],
    save_path: str):
    fieldnames = ['iteration', 'node_id', 'accuracy', 'test_loss', 'f1score', 'precision', 'recall', 'accuracy_per_0', 'accuracy_per_1',
                  'accuracy_per_2', 'accuracy_per_3', 'accuracy_per_4', 'accuracy_per_5', 'accuracy_per_6', 'accuracy_per_7',
                  'accuracy_per_8', 'accuracy_per_9']
    with open(save_path, "a", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for iteration, iteration_result in data.items():
            for node, nodes_results in iteration_result.items():
                row = copy.deepcopy(nodes_results)
                row['iteration'] = iteration
                row['node_id'] = node
                writer.writerow(row)


def save_full_contribution(
    data: dict[dict],
    save_path: str):
    fieldnames = ['node_id', 'accuracy', 'test_loss', 'f1score', 'precision', 'recall', 'accuracy_per_0', 'accuracy_per_1',
                  'accuracy_per_2', 'accuracy_per_3', 'accuracy_per_4', 'accuracy_per_5', 'accuracy_per_6', 'accuracy_per_7',
                  'accuracy_per_8', 'accuracy_per_9']
    with open(save_path, "a", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for node, nodes_results in data.items():
            row = copy.deepcopy(nodes_results)
            row['node_id'] = node
            writer.writerow(row)