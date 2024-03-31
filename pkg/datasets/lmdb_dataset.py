import copy
import logging
import os
import pickle
from collections.abc import Sequence
from functools import lru_cache
from typing import (Any, Callable, Dict, Hashable, Literal, Optional, Tuple,
                    Union)

import lmdb
import networkx as nx
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from ..utils.mixin import IFile

IndexType = Union[slice, Tensor, np.ndarray, Sequence]


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
    ):
        super(LMDBDataset, self).__init__()

        self._indices: Optional[Sequence] = None
        self._data: Optional[Dict[Hashable, Dict[Hashable, Any]]] = None

        assert source in ("raw", "others")
        self.source = source
        self.readonly = readonly
        self.dynamic = dynamic

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
        if self.source == "raw":
            self.raw_txn.abort()
            self.raw_env.close()
        if not self.dynamic and not self.readonly:
            self.processed_txn.abort()
            self.processed_env.close()

    @classmethod
    def readonly_raw(cls, raw_dir: str, raw_fname: str):
        return cls(raw_dir=raw_dir, raw_fname=raw_fname, source="raw", readonly=True)

    @classmethod
    def override_raw(cls, raw_dir: str, raw_fname: str):
        return cls(
            raw_dir=raw_dir,
            raw_fname=raw_fname,
            processed_dir=raw_dir,
            processed_fname=raw_fname,
            source="raw",
        )

    @classmethod
    def static_from_raw(
        cls,
        raw_dir: str,
        raw_fname: str,
        processed_dir: str,
        processed_fname: str,
        forced_process: bool,
    ):
        return cls(
            raw_dir=raw_dir,
            raw_fname=raw_fname,
            processed_dir=processed_dir,
            processed_fname=processed_fname,
            source="raw",
            forced_process=forced_process,
        )

    @classmethod
    def dynamic_from_raw(
        cls,
        raw_dir: str,
        raw_fname: str,
    ):
        return cls(
            raw_dir=raw_dir,
            raw_fname=raw_fname,
            source="raw",
            dynamic=True,
        )

    @classmethod
    def static_from_others(
        cls,
        dataset: Dataset,
        processed_dir: str,
        processed_fname: str,
        forced_process: bool,
    ):
        return cls(
            source_dataset=dataset,
            processed_dir=processed_dir,
            processed_fname=processed_fname,
            source="others",
            forced_process=forced_process,
        )

    @classmethod
    def dynamic_from_others(cls, dataset: Dataset):
        return cls(
            source_dataset=dataset,
            source="others",
            dynamic=True,
        )
    
    @classmethod
    def update_process_fn(cls, process_fn: Callable[[Dict[Hashable, Any]], Dict[Hashable, Any]]):
        r"""Passing a static process function to the class"""
        class NewCls(cls):
            def process(self, sample: Dict[Hashable, Any]) -> Dict[Hashable, Any]:
                return process_fn(sample)
        return NewCls

    @staticmethod
    def get_txn_len(txn: lmdb.Transaction) -> int:
        return len(list(txn.cursor().iternext(values=False)))

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
        if readonly:
            return self.raw_env, self.raw_txn
        if not to_process and not dynamic:
            env, txn = self.load(self.processed_fpath, write=False)
            return env, txn
        elif self.source == "raw":
            self._data = self._process_raw()
        else:
            self._data = self._process_others()

        if dynamic:
            return None, None
        else:
            self.save(self.processed_fpath)
            env, txn = self.load(self.processed_fpath, write=False)
            return env, txn

    def _process_raw(self) -> Dict[Hashable, Any]:
        # TODO: Check sample format
        # TODO for 2 _process: allow for None output
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

    def process(self, sample: Dict[Hashable, Any]) -> Dict[Hashable, Any]:
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
            map_size=1099511627776,
        )
        txn = env.begin(write=write)
        return env, txn

    def save(self, fpath: str) -> None:
        r"""Saving the data stored by the class"""
        if not hasattr(self, "_data"):
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

    def split_with_condition(self):
        r"""This method can be currently implemented with `to_list` and `index_select`"""
        pass
