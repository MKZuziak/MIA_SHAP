from fedata.hub.generate_dataset import generate_dataset
import os

def main():
    data_config = {
    "dataset_name" : "cifar10",
    "split_type" : "missing_classes_clustered",
    "shards": 8,
    "local_test_size": 0.3,
    "transformations": {},
    "imbalanced_clients": {},
    "save_dataset": True,
    "save_transformations": True,
    "save_blueprint": True,
    "agents": 8,
    "shuffle": True,
    "alpha": 1,
    "allow_replace": True,
    "save_path": os.getcwd(),
    "agents_cluster_belonging":
        {
            1: [0],
            2: [1],
            3: [2],
            4: [3],
            5: [4],
            6: [5],
            7: [6],
            8: [7]
        }, # Every seperate client has a its own separate 'cluster'
    "missing_classes":
        {
            1: [3, 4, 5, 6, 7, 8, 9], # classes 0, 1, 2 present only
            2: [0, 4, 5, 6, 7, 8, 9], # classes 1, 2, 3 present only
            3: [0, 3, 5, 6, 7, 8, 9], # classes 1, 2, 4 present only
            4: [0, 3, 4, 6, 7, 8, 9], # classes 1, 2, 5 present only
            5: [0, 3, 4, 5, 7, 8, 9], # classes 1, 2, 6 present only
            6: [0, 3, 4, 5, 6, 8, 9], # classes 1, 2, 7 present only
            7: [0, 3, 4, 5, 6, 7, 9], # classes 1, 2, 8 present only
            8: [0, 3, 4, 5, 6, 7, 8] # classes 1, 2, 9 present only
        } # Classes 0 and 1 are shared, the rest 
    }
    generate_dataset(config=data_config)


if __name__ == "__main__":
    main()