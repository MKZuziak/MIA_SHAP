from fedata.hub.generate_dataset import generate_dataset
import os

def main():
    data_config = {
    "dataset_name" : "mnist",
    "split_type" : "homogeneous",
    "shards": 4,
    "local_test_size": 0.3,
    "transformations": {0: {"transformation_type": "noise", "noise_multiplyer": 1.5},
                        1: {"transformation_type": "noise", "noise_multiplyer": 0.09},
                        2: {"transformation_type": "noise", "noise_multiplyer": 0.01}},
    "imbalanced_clients": {},
    "save_dataset": True,
    "save_transformations": True,
    "save_blueprint": True,
    "agents": 4,
    "shuffle": True,
    "save_path": os.getcwd()}
    generate_dataset(config=data_config)


if __name__ == "__main__":
    main()