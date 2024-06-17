from fedata.hub.generate_dataset import generate_dataset
import os

def main():
    data_config = {
    "dataset_name" : "cifar10",
    "split_type" : "dirchlet_clusters_nooverlap",
    "shards": 12,
    "local_test_size": 0.3,
    "transformations": {},
    "imbalanced_clients": {},
    "save_dataset": True,
    "save_transformations": True,
    "save_blueprint": True,
    "agents": 12,
    "shuffle": True,
    "alpha": 1,
    "allow_replace": True,
    "save_path": os.getcwd(),
    "agents_cluster_belonging":
        {
            1: [0, 1, 2, 3],
            2: [4, 5, 6, 7],
            3: [8, 9, 10, 11]
        },
    "missing_classes":
        {
            1: [0, 1, 2],
            2: [3, 4, 5],
            3: [6, 7, 8, 9]
        }
    }
    generate_dataset(config=data_config)


if __name__ == "__main__":
    main()