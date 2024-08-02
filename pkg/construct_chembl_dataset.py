from datasets.chembl_dataset import ChemBLActivityDataset

if __name__ == "__main__":
    dataset = ChemBLActivityDataset(target_name="Epoxide hydratase")
    # dataset = ChemBLActivityDataset(target_name="Carbonic anhydrase II")
    # dataset = ChemBLActivityDataset(target_name="Carbonic anhydrase XII")
    print(dataset[0])
    print(len(dataset))
