import os

import datasets
import numpy as np
import pandas as pd
from torchvision import transforms


def save_blueprint(
    dataset: list[datasets.arrow_dataset.Dataset, list[datasets.arrow_dataset.Dataset]], 
    path:str,
    blueprine_name:str
    ):
    name = os.path.join(path, (blueprine_name + ".csv"))
    with open(name, 'a+', ) as csv_file:
        orchestrator_data = dataset[0]
        nodes_data = dataset[1]
        
        # WRITE HEADERS
        header = ["client_id", "partition", "total_samples"]
        labels = ['0', '1', '2', '3', '4', '5', '6', '7']
        header.extend(labels)
        csv_file.write(",".join(header) + '\n')
        
        translation = False
        try:
            labels = [int(label) for label in labels]
        except ValueError:
            translation = True
            labels_translation = {integer: label for (integer, label) in zip(range(len(labels)), labels)}
            labels = [value for value in labels_translation.keys()]

        # WRITE ORCHESTRATOR
        row = ['orchestrator', 'central_test_set', str(len(orchestrator_data))]
        for label in labels:
            row.append(str(len(orchestrator_data.filter(lambda inst: inst['label'] == label))))
        csv_file.write(",".join(row) + '\n')

        # WRITE CLIENTS
        for client, data in enumerate(nodes_data):
            row = [str(client), 'train_set', str(len(data[0]))]
            for label in labels:
                row.append(str(len(data[0].filter(lambda inst: inst['label'] == label))))
            csv_file.write(",".join(row) + '\n')

            row = [str(client), 'test_set', str(len(data[1]))]
            for label in labels:
                row.append(str(len(data[1].filter(lambda inst: inst['label'] == label))))
            csv_file.write(",".join(row) + '\n')
        
    #Optional: write labels (if translated)
    if translation:
        name = os.path.join(path, (blueprine_name + "translation" + ".txt"))
        with open(name, "w+") as txt_file:
            for integer, label in labels_translation.items():
                txt_file.write(f"{str(integer)}: {label}")


def save_dataset(
    path: str,
    name: str,
    loaded_dataset: int
    ):
    n = os.path.join(path, name)
    dataset_length = len(loaded_dataset[1])
    dataset = {"orchestrator_test_set": loaded_dataset[0]}
    for client in range(dataset_length):
        dataset[f"client_{client}_train_set"] = loaded_dataset[1][client][0]
        dataset[f"client_{client}_test_set"] = loaded_dataset[1][client][1]
    print(f"The dataset was packed and will be save in format: {dataset}")
    dataset = datasets.DatasetDict(dataset)
    dataset.save_to_disk(n)
    
    
def convert_to_hg(
    dataset: list[dict]
):
    dataset = [{'image': dataset[index][0], 'label': int(dataset[index][1][0])} for index in range(len(dataset))]
    dataset = datasets.Dataset.from_list(dataset)
    return dataset


def create_dirchlet_matrix(
    alpha: int | float,
    size: int,
    k: int
    ):
    generator = np.random.default_rng()
    alpha = np.full(k, alpha)
    s = generator.dirichlet(alpha, size)
    return s


def homogeneous(
    dataset: datasets.arrow_dataset.Dataset,
    shards: int,
    seed: int = 42,
    shuffle: bool = False,
    local_test_size: float = 0.2
    ):
    # DATASET SHUFFLING
    if shuffle == True:
        dataset = dataset.shuffle(seed = seed)
    # SHARD PARTITIONING
    nodes_data = []
    for shard in range(shards): # Each shard corresponds to one agent.
        agent_data = dataset.shard(num_shards=shards, index=shard)
        # In-shard split between test and train data.
        agent_data = agent_data.train_test_split(test_size=local_test_size)
        nodes_data.append([agent_data['train'], agent_data['test']])
    # RETURNING NODES DATA
    return nodes_data


def dirchlet(
    dataset: datasets.arrow_dataset.Dataset,
    shards: int,
    alpha: float,
    num_classes: int,
    seed: int = 42,
    shuffle: bool = False,
    local_test_size: float = 0.2
    ):
    #DATASET SHUFFLING
    if shuffle == True:
        dataset = dataset.shuffle(seed = seed)
    d_matrix = create_dirchlet_matrix(
        alpha=alpha,
        size=shards,
        k=num_classes
        )
    avg_size = int(np.floor(dataset.num_rows / shards))
    pandas_df = dataset.to_pandas().drop('image', axis=1)
    nodes_data = []
    
    for agent in range(d_matrix.shape[0]):
        sampling_weights = pd.Series({label: p for (label, p) in zip(range(num_classes), d_matrix[agent])})
        pandas_df["weights"] = pandas_df['label'].apply(lambda x: sampling_weights[x])
        sample = pandas_df.sample(n = avg_size, weights='weights', random_state=seed) # allow replace is False be default
        sampled_data = dataset.select(sample.index)
        # Removing samples indexes
        pandas_df.drop(sample.index, inplace=True)
        agent_data = sampled_data.train_test_split(test_size=local_test_size)
        nodes_data.append([agent_data['train'], agent_data['test']])
    return nodes_data


def missing_classes_clustered(
    dataset: datasets.arrow_dataset.Dataset,
    shards: int,
    agents_cluster_belonging: dict,
    missing_classes_dict: dict,
    seed: int = 42,
    shuffle: bool = False,
    local_test_size: float = 0.2
    ):
        nodes_data = []
        if shuffle == True:
            dataset = dataset.shuffle(seed = seed)
        no_agents = shards
        pandas_df = dataset.to_pandas().drop('image', axis=1)
        labels = sorted(pandas_df.label.unique())
        sample_size = int(len(dataset) / no_agents)
                
        for cluster, agents in agents_cluster_belonging.items():
            random_state = seed # first random state
            # 1. Identifying missing classes 
            missing_classes = missing_classes_dict[cluster]
            # 2. Sampling indexes
            try:
                sampling_weights = {key: (1 / ( len(labels) - len(missing_classes)))
                                if key not in missing_classes else 0 for key in labels}
            except Exception as e:
                print(e)
            # 3. Applying weights
            pandas_df["weights"] = pandas_df['label'].apply(lambda x: sampling_weights[x])
            for _ in agents:
                sample = pandas_df.sample(n = sample_size, weights='weights', random_state=random_state, replace=True)
                # 4. Selecting indexes and performing test - train split
                sampled_data = dataset.filter(lambda _, idx: idx in list(sample.index), with_indices=True)
                agent_data = sampled_data.train_test_split(test_size=local_test_size)
                nodes_data.append([agent_data['train'], agent_data['test']])
                random_state += 100
        return nodes_data


# Data transformation func
data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])


