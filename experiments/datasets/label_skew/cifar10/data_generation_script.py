from fedata.hub.generate_dataset import generate_dataset
import os

def main():
    data_config = {
    "dataset_name" : "cifar10",
    "split_type" : "dirchlet_clusters_nooverlap",
    "shards": 6,
    "local_test_size": 0.3,
    "transformations": {},
    "imbalanced_clients": {},
    "save_dataset": True,
    "save_transformations": True,
    "save_blueprint": True,
    "agents": 6,
    "shuffle": True,
    "alpha": 0.5,
    "allow_replace": True,
    "save_path": os.getcwd(),
    "agents_cluster_belonging":
        {
            0: [0, 1, 2, 3],
            1: [5, 6]
        },
    "missing_classes":
        {
            0: [1],
            1: [8, 9]
        }
    }
    generate_dataset(config=data_config)


if __name__ == "__main__":
    main()