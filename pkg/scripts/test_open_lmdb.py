import os

from ..datasets.lmdb_dataset import LMDBDataset

path = "/data03/gtguo/data/chembl/lmdb/target_hits/sEH/sEH_active_thr6.0.lmdb"
dataset = LMDBDataset.readonly_raw(*os.path.split(path))
print(dataset[0])
print(len(dataset))
