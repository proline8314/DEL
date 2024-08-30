import copy
import gc
import logging
import os
import pickle
import sys
from collections.abc import Sequence
from functools import lru_cache
from multiprocessing import Pool
from typing import (Any, Callable, Dict, Hashable, Literal, NewType, Optional,
                    Tuple, Union)

import lmdb
import networkx as nx
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from ..utils.mixin import IFile

IndexType = Union[slice, Tensor, np.ndarray, Sequence]
Keys = Union[Hashable, Tuple[Hashable]]


class LMDBDataset(Dataset, IFile):
    r"""A dataset that reads and writes lmdb files.
    If `readonly` or `dynamic`, only raw file path is needed, elsewhere processed file path should also be assigned.
    `forced_process` serves for the case when the processed file is expected to be updated.
    """

    # TODO : update file handling procedure according to the note
    def __init__(
        self,
        *,
        raw_dir: str = None,
        raw_fname: str = None,
        processed_dir: str = None,
        processed_fname: str = None,
        source_dataset: Dataset = None,
        source: Literal["raw", "others"] = "raw",
        readonly: bool = False,
        dynamic: bool = False,
        forced_process: bool = False,
        **kwargs,
    ):
        super(LMDBDataset, self).__init__()

        self._indices: Optional[Sequence] = None
        self._data: Optional[Dict[Hashable, Any]] = None

        assert source in ("raw", "others")
        self.source = source
        self.readonly = readonly
        self.dynamic = dynamic

        self.map_size = (
            1099511627776 if "map_size" not in kwargs.keys() else kwargs["map_size"]
        )
        self.nprocs = 1 if "nprocs" not in kwargs.keys() else kwargs["nprocs"]
        self.process = (
            self.process
            if "process_fn" not in kwargs.keys()
            else copy.deepcopy(kwargs["process_fn"])
        )

        if self.source == "raw":
            self.raw_dir = raw_dir
            self.raw_fname = raw_fname
            self.raw_fpath = os.path.join(self.raw_dir, self.raw_fname)
            self.raw_env, self.raw_txn = self.load(self.raw_fpath, write=False)
        elif self.source == "others":
            self.source_dataset = source_dataset
        else:
            raise ValueError(
                f'The parameter `source` can only be "raw" or "others", but here it is "{source}".'
            )

        to_process = False
        if not readonly and not dynamic:
            self.processed_dir = processed_dir
            self.processed_fname = processed_fname
            self.processed_fpath = os.path.join(
                self.processed_dir, self.processed_fname
            )
            to_process = forced_process or (
                not self._check_processed(self.processed_fpath)
                or not self._check_processed_complete(self.processed_fpath)
            )
        self.processed_env, self.processed_txn = self._assign_txn(
            readonly, dynamic, to_process
        )

    def __len__(self) -> int:
        # TODO : reduce coupling
        # TODO refresh
        # !? This means that the dataset is not free of adding new data
        if not hasattr(self, "_len"):
            if self._indices is not None:
                self._len = len(self._indices)
            elif self.readonly:
                self._len = self.get_txn_len(self.raw_txn)
            elif self.dynamic:
                self._len = len(self._data)
            else:
                self._len = self.get_txn_len(self.processed_txn)
        return self._len

    def __getitem__(
        self, idx: Union[int, np.integer, IndexType]
    ) -> Union[Dict[Hashable, Any], Any]:
        if (
            isinstance(idx, (int, np.integer))
            or (isinstance(idx, Tensor) and idx.dim() == 0)
            or (isinstance(idx, np.ndarray) and np.isscalar(idx))
        ):

            data = self.get_fn(self.indices[idx])
            return data

        else:
            return self.index_select(idx)

    def __del__(self):
        # TODO following bug
        """
        Exception ignored in: <function LMDBDataset.__del__ at 0x7f36fec928c0>
        Traceback (most recent call last):
          File "/data02/gtguo/DEL/pkg/datasets/lmdb_dataset.py", line 111, in __del__
            if self.source == "raw":
        AttributeError: 'ChemBLActivityDataset' object has no attribute 'source'
        """
        if self.source == "raw":
            self.raw_txn.abort()
            self.raw_env.close()
        if not self.dynamic and not self.readonly:
            self.processed_txn.abort()
            self.processed_env.close()

    @classmethod
    def readonly_raw(cls, raw_dir: str, raw_fname: str, **kwargs):
        return cls(
            raw_dir=raw_dir, raw_fname=raw_fname, source="raw", readonly=True, **kwargs
        )

    @classmethod
    def override_raw(cls, raw_dir: str, raw_fname: str, **kwargs):
        return cls(
            raw_dir=raw_dir,
            raw_fname=raw_fname,
            processed_dir=raw_dir,
            processed_fname=raw_fname,
            source="raw",
            forced_process=True,
            **kwargs,
        )

    @classmethod
    def static_from_raw(
        cls,
        raw_dir: str,
        raw_fname: str,
        processed_dir: str,
        processed_fname: str,
        forced_process: bool = False,
        **kwargs,
    ):
        return cls(
            raw_dir=raw_dir,
            raw_fname=raw_fname,
            processed_dir=processed_dir,
            processed_fname=processed_fname,
            source="raw",
            forced_process=forced_process,
            **kwargs,
        )

    @classmethod
    def dynamic_from_raw(cls, raw_dir: str, raw_fname: str, **kwargs):
        return cls(
            raw_dir=raw_dir, raw_fname=raw_fname, source="raw", dynamic=True, **kwargs
        )

    @classmethod
    def static_from_others(
        cls,
        dataset: Dataset,
        processed_dir: str,
        processed_fname: str,
        forced_process: bool = False,
        **kwargs,
    ):
        # ! Fix the bug
        """
        Traceback (most recent call last):
        File "/data02/gtguo/DEL/pkg/infer_ef_ca9.py", line 159, in <module>
            active_dataset = active_dataset.static_from_others(
        File "/data02/gtguo/DEL/pkg/datasets/lmdb_dataset.py", line 181, in static_from_others
            return cls(
        File "/data02/gtguo/DEL/pkg/datasets/lmdb_dataset.py", line 77, in __init__
            self.processed_env, self.processed_txn = self._assign_txn(
        File "/data02/gtguo/DEL/pkg/datasets/lmdb_dataset.py", line 276, in _assign_txn
            self._data = self._process_others()
        File "/data02/gtguo/DEL/pkg/datasets/lmdb_dataset.py", line 302, in _process_others
            sample = self.source_dataset[idx]
        File "/data02/gtguo/DEL/pkg/datasets/lmdb_dataset.py", line 105, in __getitem__
            data = self.get_fn(self.indices[idx])
        File "/data02/gtguo/DEL/pkg/datasets/lmdb_dataset.py", line 335, in _static_get
            return pickle.loads(self.processed_txn.get(f"{index}".encode()))
        lmdb.Error: Attempt to operate on closed/deleted/dropped object.
        Exception ignored in: <function LMDBDataset.__del__ at 0x7f95f4535750>
        Traceback (most recent call last):
        File "/data02/gtguo/DEL/pkg/datasets/lmdb_dataset.py", line 124, in __del__
            self.processed_txn.abort()
        AttributeError: 'NewCls' object has no attribute 'processed_txn'
        """

        return cls(
            source_dataset=dataset,
            processed_dir=processed_dir,
            processed_fname=processed_fname,
            source="others",
            forced_process=forced_process,
            **kwargs,
        )

    @classmethod
    def dynamic_from_others(cls, dataset: Dataset, **kwargs):
        return cls(source_dataset=dataset, source="others", dynamic=True, **kwargs)

    @classmethod
    def update_process_fn(
        cls, process_fn: Callable[[Dict[Hashable, Any]], Dict[Hashable, Any]]
    ):
        r"""Passing a static process function to the class"""

        class NewCls(cls):
            @staticmethod
            def process(sample: Dict[Hashable, Any]) -> Dict[Hashable, Any]:
                return process_fn(sample)

        return NewCls

    @classmethod
    def append_process_fn(cls, process_fn: Callable[[Any], Any]):
        r"""Appending a process function to the class"""
        # TODO: ! test
        old_process_fn = cls.process

        class NewCls(cls):
            def process(self, sample: Any) -> Any:
                return process_fn(old_process_fn(self, sample))

        return NewCls

    @staticmethod
    def get_txn_len(txn: lmdb.Transaction) -> int:
        return len(list(txn.cursor().iternext(values=False)))

    def _check_meta_info(self):
        # TODO: use meta info to check the validity of the dataset, reload if not valid
        return True

    def _check_processed(self, fpath: str):
        return os.path.exists(fpath)

    def _check_processed_complete(self, processed_fpath: str):
        r"""Check if the processed file is complete, in case the processing was interrupted during the last run.
        Since the destructor will take care of the cleanup of the incomplete file, simply check if the file has any data.
        """
        env, txn = self.load(processed_fpath, write=False)
        complete = self.get_txn_len(txn) != 0
        env.close()
        return complete

    def _check_has_key(
        self,
        key: Union[Hashable, Tuple[Hashable]],
        _sample: Optional[Dict[Hashable, Any]] = None,
    ) -> bool:
        # TODO support to list
        _sample = self.get_fn(0) if _sample is None else _sample
        if not hasattr(_sample, "keys"):
            return False
        elif not isinstance(key, tuple):
            return key in _sample.keys()
        elif len(key) == 1:
            key = key[0]
            return key in _sample.keys()
        else:
            key, *rest = key
            flag = key in _sample.keys()
            _sample = _sample[key]
            return flag and self._check_has_key(tuple(rest), _sample)

    def _check_sample_format_unified(self) -> bool:
        r"""Check if the sample format is unified across the dataset"""
        # TODO
        return True

    def _assign_txn(
        self, readonly: bool, dynamic: bool, to_process: bool
    ) -> Union[Tuple[lmdb.Environment, lmdb.Transaction], Tuple[None, None]]:
        # TODO async handling & saving
        if readonly:
            return self.raw_env, self.raw_txn
        if not to_process and not dynamic:
            env, txn = self.load(self.processed_fpath, write=False)
            return env, txn
        elif self.source == "raw":
            if self.nprocs > 1:
                self._data = self._process_raw_multiprocessing()
            else:
                self._data = self._process_raw()
        else:
            if self.nprocs > 1:
                self._data = self._process_others_multiprocessing()
            else:
                self._data = self._process_others()

        if dynamic:
            return None, None
        else:
            self.save(self.processed_fpath)
            env, txn = self.load(self.processed_fpath, write=False)
            self._data = None  # release mem
            return env, txn

    def _process_raw(self) -> Dict[Hashable, Any]:
        # TODO: Check sample format
        # TODO for 2 _process: allow for None output (or other placeholder representing for skipping the sample)
        # TODO: multiprocessing
        data = {}
        logging.info("processing dataset...")
        for idx in tqdm(range(self.get_txn_len(self.raw_txn))):
            sample = pickle.loads(self.raw_txn.get(f"{idx}".encode()))
            sample = self.process(sample)
            data[idx] = sample
        return data

    def _process_others(self) -> Dict[Hashable, Any]:
        data = {}
        logging.info("processing dataset...")
        for idx in tqdm(range(len(self.source_dataset))):
            sample = self.source_dataset[idx]
            sample = self.process(sample)
            data[idx] = sample
        return data

    def _process_raw_multiprocessing(self) -> Dict[Hashable, Any]:
        data = {}

        global copy_process
        copy_process = copy.copy(self.process)

        # TODO ! magic number
        # TODO wrap in a function
        # ? Integrity of self._data for dynamic processing
        # ? Here the saving process circumvents the assign_txn function

        maximum_dataset_slice = 10_000
        # maximum_dataset_slice = 100_000
        range_generators = [
            range(
                maximum_dataset_slice * i,
                min(maximum_dataset_slice * (i + 1), self.get_txn_len(self.raw_txn)),
            )
            for i in range(
                np.ceil(self.get_txn_len(self.raw_txn) / maximum_dataset_slice).astype(
                    int
                )
            )
        ]

        if len(range_generators) > 1 and self.dynamic:
            logging.warning(
                "The dataset is too large to be processed in one slice, dynamic processing is not supported."
            )
            sys.exit(1)

        logging.info("processing dataset with multiprocessing...")
        logging.info(f"nprocs: {self.nprocs}")
        logging.info(f"total slices: {len(range_generators)}")

        for outer_idx, range_generator in enumerate(range_generators):
            with Pool(self.nprocs) as pool:
                for idx, sample in tqdm(
                    enumerate(
                        pool.imap(
                            copy_process,
                            (
                                self._get_from_txn(self.raw_txn, i)
                                for i in range_generator
                            ),
                            chunksize=100,
                        )
                    ),
                    total=len(range_generator),
                ):
                    data_idx = idx + maximum_dataset_slice * outer_idx
                    data[data_idx] = sample

            # logging.info(f"Hit the maximum dataset slice: {maximum_dataset_slice}")

            self._data = data
            self.save(self.processed_fpath)

            # ! release memory
            self._data = None
            del data
            gc.collect()
            data = {}
            
        return data

    def _process_others_multiprocessing(self) -> Dict[Hashable, Any]:
        # TODO multiprocessing update
        data = {}
        logging.info("processing dataset...")
        with Pool(self.nprocs) as pool:
            for idx, sample in tqdm(
                enumerate(
                    pool.imap(
                        self.process,
                        self.source_dataset,
                        chunksize=100,
                    )
                ),
                total=len(self.source_dataset),
            ):
                data[idx] = sample
        return data

    @staticmethod
    def process(sample: Dict[Hashable, Any]) -> Any:
        r"""Users override this method to achieve custom functionality"""
        return sample

    def _get_with_keys(
        self,
        key: Union[Hashable, Tuple[Hashable]],
        idx: int,
        _sample: Optional[Dict[Hashable, Any]] = None,
    ):
        r"""Get the data with the given keys"""
        # Working rely on a unified sample format
        if _sample is None:
            _sample = self.get_fn(idx)

        if not isinstance(key, tuple):
            return _sample[key]
        elif len(key) == 1:
            key = key[0]
            return _sample[key]
        else:
            key, *rest = key
            _sample = _sample[key]
            return self._get_with_keys(tuple(rest), idx, _sample)

    @lru_cache(maxsize=16)
    def _static_get(self, index):
        # TODO: maybe open for transformation plug-ins here or in `__getitem__`
        return pickle.loads(self.processed_txn.get(f"{index}".encode()))

    @lru_cache(maxsize=16)
    def _dynamic_get(self, index):
        return self._data[index]

    @lru_cache(maxsize=16)
    def _get_from_txn(self, txn: lmdb.Transaction, index: int):
        return pickle.loads(txn.get(f"{index}".encode()))

    @property
    def get_fn(self) -> Any:
        # TODO refresh
        if not hasattr(self, "_cached_get_fn"):
            self._cached_get_fn = (
                self._dynamic_get if self.dynamic else self._static_get
            )
        return self._cached_get_fn

    @property
    def indices(self) -> Sequence:
        return range(len(self)) if self._indices is None else self._indices

    def load(
        self, fpath: str, write: bool, replace: bool = False
    ) -> Tuple[lmdb.Environment, lmdb.Transaction]:
        r"""Loading the data from the file path, and returning the environment and transaction object"""
        # ? Static for now
        if replace and os.path.exists(fpath):
            os.remove(fpath)

        env = lmdb.open(
            fpath,
            subdir=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=self.map_size,
        )
        txn = env.begin(write=write)
        return env, txn

    def save(self, fpath: str) -> None:
        r"""Saving the data stored by the class"""
        if not hasattr(self, "_data") or self._data is None:
            raise AttributeError("No data to save")

        fdir, _ = os.path.split(fpath)
        os.makedirs(fdir, exist_ok=True)
        _env, _txn = self.load(fpath, write=True)
        for key, value in self._data.items():
            _txn.put(f"{key}".encode(), pickle.dumps(value, protocol=-1))
        _txn.commit()
        _env.close()

    def index_select(self, idx: IndexType) -> "Dataset":
        indices = self.indices

        if isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step
            # Allow floating-point slicing, e.g., dataset[:0.9]
            if isinstance(start, float):
                start = round(start * len(self))
            if isinstance(stop, float):
                stop = round(stop * len(self))
            idx = slice(start, stop, step)

            indices = indices[idx]

        elif isinstance(idx, Tensor) and idx.dtype == torch.long:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Tensor) and idx.dtype == torch.bool:
            idx = idx.flatten().nonzero(as_tuple=False)
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == bool:
            idx = idx.flatten().nonzero()[0]
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            indices = [indices[i] for i in idx]

        else:
            raise IndexError(
                f"Only slices (':'), list, tuples, torch.tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')"
            )

        dataset = copy.copy(self)
        dataset._indices = indices
        dataset._len = len(indices)
        return dataset

    def shuffle(
        self,
        return_perm: bool = False,
    ) -> Union["Dataset", Tuple["Dataset", Tensor]]:
        perm = torch.randperm(len(self))
        dataset = self.index_select(perm)
        return (dataset, perm) if return_perm is True else dataset

    def to_list(self, key: Union[Hashable, Tuple[Hashable]]) -> list:
        r"""Convert the dataset to a list given a key"""
        if not self._check_has_key(key):
            raise KeyError(f"Key {key} not found in the dataset")
        return [self._get_with_keys(key, idx) for idx in self.indices]

    def to_array(self, key: Union[Hashable, Tuple[Hashable]]) -> np.ndarray:
        r"""Convert the dataset to a numpy array given a key"""
        return np.array(self.to_list(key))

    def split_with_condition(
        self,
        condition: Callable[[Dict[Hashable, Any]], bool],
        argument: Tuple[Keys],
        save: bool = False,
        load: bool = False,
        *,
        true_dataset_name: str = None,
        false_dataset_name: str = None,
        **kwargs,
    ) -> Tuple["Dataset", "Dataset"]:
        r"""
        Split the dataset based on the condition.

        Example:
        ```
        def condition(label):
            return label == 1
        true_dataset, false_dataset = dataset.split_with_condition(condition, ("label",), save=True)
        ```python

        """
        # TODO (maybe): support multiple conditions
        # TODO: clean up with dataset name part
        # TODO: sample itself? more types of input?
        if (
            load
            and os.path.exists(
                os.path.join(self.processed_dir, f"{true_dataset_name}.lmdb")
            )
            and os.path.exists(
                os.path.join(self.processed_dir, f"{false_dataset_name}.lmdb")
            )
        ):
            true_dataset = LMDBDataset.readonly_raw(
                self.processed_dir, f"{true_dataset_name}.lmdb"
            )
            false_dataset = LMDBDataset.readonly_raw(
                self.processed_dir, f"{false_dataset_name}.lmdb"
            )
            return true_dataset, false_dataset

        # TODO: test with numpy array method
        # ! save and load error
        true_indices = []
        false_indices = []
        for idx in tqdm(self.indices, desc="Splitting dataset"):
            args = [self._get_with_keys(key, idx) for key in argument]
            if condition(*args, **kwargs):
                true_indices.append(idx)
            else:
                false_indices.append(idx)
        true_dataset = self.index_select(true_indices)
        false_dataset = self.index_select(false_indices)
        if save:
            true_dataset.save(
                os.path.join(self.processed_dir, f"{true_dataset_name}.lmdb")
            )
            false_dataset.save(
                os.path.join(self.processed_dir, f"{false_dataset_name}.lmdb")
            )
        return true_dataset, false_dataset
    
    def split_with_idx(self, idx: IndexType) -> Tuple["Dataset", "Dataset"]:
        r"""Split the dataset with the given indices"""
        return self.index_select(idx), self.index_select(
            [i for i in self.indices if i not in idx]
        )
