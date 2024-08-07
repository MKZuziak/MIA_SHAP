from fedata.hub.generate_dataset import generate_dataset
import os

def main():
    data_config = {
    "dataset_name" : "mnist",
    "split_type" : "homogeneous",
    "shards": 8,
    "local_test_size": 0.3,
    "transformations": {0: {"transformation_type": "noise", "noise_multiplyer": 0.002},
                        1: {"transformation_type": "noise", "noise_multiplyer": 0.005},
                        2: {"transformation_type": "noise", "noise_multiplyer": 0.005},
                        3: {"transformation_type": "noise", "noise_multiplyer": 0.01}
                        },
    "imbalanced_clients": {},
    "save_dataset": True,
    "save_transformations": True,
    "save_blueprint": True,
    "agents": 8,
    "shuffle": True,
    "save_path": os.getcwd()}
    generate_dataset(config=data_config)


if __name__ == "__main__":
    main()