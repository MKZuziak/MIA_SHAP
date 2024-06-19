from fedata.hub.generate_dataset import generate_dataset
import os

def main():
    data_config = {
    "dataset_name" : "fmnist",
    "split_type" : "dirchlet_clusters_nooverlap",
    "shards": 10,
    "local_test_size": 0.3,
    "transformations": {},
    "imbalanced_clients": {},
    "save_dataset": True,
    "save_transformations": True,
    "save_blueprint": True,
    "agents": 10,
    "shuffle": True,
    "alpha": 1,
    "allow_replace": True,
    "save_path": os.getcwd(),
    "agents_cluster_belonging":
        {
            1: [0, 1, 2],
            2: [3, 4, 5],
            3: [6, 7, 8, 9]
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