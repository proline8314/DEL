import os

from ..datasets.lmdb_dataset import LMDBDataset

path = "/data03/gtguo/data/chemdiv/lmdb/chemdiv.lmdb"
dataset = LMDBDataset.readonly_raw(*os.path.split(path))
print(dataset[0])
print(len(dataset))
