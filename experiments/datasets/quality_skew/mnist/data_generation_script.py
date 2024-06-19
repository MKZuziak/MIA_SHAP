from fedata.hub.generate_dataset import generate_dataset
import os

def main():
    data_config = {
    "dataset_name" : "mnist",
    "split_type" : "homogeneous",
    "shards": 10,
    "local_test_size": 0.3,
    "transformations": {0: {"transformation_type": "noise", "noise_multiplyer": 0.01},
                        1: {"transformation_type": "noise", "noise_multiplyer": 0.05},
                        2: {"transformation_type": "noise", "noise_multiplyer": 0.15},
                        3: {"transformation_type": "noise", "noise_multiplyer": 0.5},
                        4: {"transformation_type": "noise", "noise_multiplyer": 1.0},
                        5: {"transformation_type": "rotation", "degrees": 270},
                        6: {"transformation_type": "perspective_change", "distortion_scale": 0.9, "transformation_probability": 0.9},
                        },
    "imbalanced_clients": {},
    "save_dataset": True,
    "save_transformations": True,
    "save_blueprint": True,
    "agents": 10,
    "shuffle": True,
    "save_path": os.getcwd()}
    generate_dataset(config=data_config)


if __name__ == "__main__":
    main()