# README

## Environment Setup

Install dependencies in `requirements_conda.txt`:

```bash
while read requirement; do conda install --yes $requirement; done < requirements_conda.txt
```

Meanwhile, the following dependencies are installed manually:

- [PyTorch](https://pytorch.org/get-started/locally/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

## Data Preparation

### LMDB Dataset Recipies

We use [lmdb](https://pypi.org/project/lmdb/) as the database to store the dataset. The implementation is in `./pkg/datasets/lmdb_dataset.py`. 

Call function `dynamic_from_raw` or `dynamic_from_others` to create an in-memory dataset, and `static_from_raw` or `static_from_others` to create a static dataset. 

Passing an integer > 1 to argument `nprocs` in the constructor will enable multiprocessing for data preparation.

Either overwriting `process` function or passing a processing function to argument `process_fn` in the constructor can be used for preparation customization. If multiprocessing is needed, the latter is recommended.

Several examples are provided in `./pkg/packing/`.

### Data Format

Each entry in the dataset for training should be stored in the following format:

```bash
{
    "bb_pyg_data": 
    {
        "x": np.array, 
        "edge_idx": np.array, 
    },

    "pyg_data": 
    {
        "x": np.array, 
        "edge_idx": np.array, 
        "edge_attr": np.array,
        "synthon_index": np.array,
    },

    "readout":
    {
        "$TARGET_NAME$":
        {
            "target": np.array,
            "control": np.array,
        },
        ...,
    }

}
```

The class `CollateDataset` in `./pkg/train.py` is used to collate the samples into pytorch-geometric graphs.

The following workflow is recommended.

#### Packing raw data

See `./pkg/packing/pack_*.py` for examples.

DEL readouts, DEL molecules SMILES, and building blocks SMILES are packed into the dataset.

#### Graph coloring

See `./pkg/packing/index_*.py` for examples.

Atom mapping is performed on the DEL molecules and building blocks w/ `./pkg/utils/get_synthon_index.py`. The results are stored in `synthon_index`.

#### Feature extraction

See `./pkg/packing/feat_*.py` for examples.

The featuring functions are stored in `./pkg/utils/mol_feat_v2.py`.

### Automatic Synthon Extraction

If synthon strucures are not provided, you may refer to `./code/data_preprocess/prep_syns_003.py` to extract synthons from the DEL molecules. 

The algorithm is based on maximum common substructure (MCS) search. Due to the missing reactive groups in the DEL molecules, the extracted synthons may be inaccurate and duplicated.

## Training

The following command is used to train the model. You can wrap it with `nohup` to run it in the background.

```bash
python -m pkg.train --collate_dataset
```

Here's some optional arguments:

- `--update_loss`: Update the parameters of the loss function during training.
- `--lr_schedule`: Use a learning rate scheduler.
- `--loss_sigma_correction`: Use the corrected ZIP loss function. Useful when non-zero readouts dominate the dataset.

## Inference

### Affinity Prediction

To measure the screening performance of the model, use the following command:

```bash
python -m pkg.scripts.refnet_v2.calc_ef
```

The positive and negative control data should be provided and passed to the script. You may refer to `./pkg/datasets/chembl_dataset.py` and `./pkg/scripts/data_handling/construct_chembl_dataset.py` to construct the positive datasets from ChEMBL.