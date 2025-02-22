from fedata.hub.generate_dataset import generate_dataset
import os

def main():
    data_config = {
    "dataset_name" : "cifar10",
    "split_type" : "dirchlet",
    "shards": 5,
    "local_test_size": 0.3,
    "transformations": {},
    "imbalanced_clients": {},
    "save_dataset": True,
    "save_transformations": True,
    "save_blueprint": True,
    "agents": 5,
    "shuffle": True,
    "alpha": 1,
    "save_path": os.getcwd()}
    generate_dataset(config=data_config)


if __name__ == "__main__":
    main()