import os

from .datasets.lmdb_dataset import LMDBDataset

fpath = ""
dataset = LMDBDataset.readonly_raw(*os.path.split(fpath))
print(dataset[0])
print(len(dataset))