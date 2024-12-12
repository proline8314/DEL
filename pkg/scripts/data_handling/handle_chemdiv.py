from ...datasets.chemdiv_dataset import ChemDivDataset
from ...datasets.lmdb_dataset import LMDBDataset

if __name__ == '__main__':
    # dataset = ChemDivDataset()
    dataset = LMDBDataset.readonly_raw(raw_dir="/data03/gtguo/data/chemdiv/lmdb", raw_fname="chemdiv.lmdb")
    print(dataset[0])
    print(len(dataset))