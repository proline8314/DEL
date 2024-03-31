import os

from rdkit import Chem
from rdkit.Chem import rdmolops

from .lmdb_dataset import LMDBDataset

DATASET_DIR = "/data02/gtguo/DEL/data/dataset/acs.jcim.2c01608"
RAW_FILE_NAME = "features.lmdb"
PROCESSED_FILE_NAME = "graph_dataset.lmdb"

class GraphDataset(LMDBDataset):
    def __init__(self, forced_reload: bool = False, target_name: str = "ca9"):
        # TODO : support for different raw datasets
        super(GraphDataset, self).__init__(raw_dir=DATASET_DIR, raw_fname=RAW_FILE_NAME, processed_dir=DATASET_DIR, processed_fname=PROCESSED_FILE_NAME, forced_process=forced_reload)

    def process(self):
        pass



