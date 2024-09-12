from datasets.graph_dataset import GraphDataset
from datasets.lmdb_dataset import LMDBDataset

dataset_dir = GraphDataset.DATASET_DIR
dataset_name = GraphDataset.name_dict['ca9']
dataset = LMDBDataset.readonly_raw(dataset_dir, dataset_name)
print(dataset[0])
print(len(dataset))
