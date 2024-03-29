import copy
import os
import pickle
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Dict, Hashable, Optional, Tuple, Union

import lmdb
import networkx as nx
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..utils.mixin import IFile

IndexType = Union[slice, Tensor, np.ndarray, Sequence]


class LMDBDataset(Dataset, IFile):
    def __init__(
        self,
        raw_dir: str,
        raw_fname: str,
        readonly: bool = False,
        dynamic: bool = False,
        forced_process: bool = False,
        *,
        processed_dir: str,
        processed_fname: str,
    ):
        super(LMDBDataset, self).__init__()
        self.raw_dir = raw_dir
        self.raw_fname = raw_fname
        self.raw_fpath = os.path.join(self.raw_dir, self.raw_fname)
        self.raw_env, self.raw_txn = self.load(self.raw_fpath, write=False)

        self.readonly = readonly
        self.dynamic = dynamic
        self._indices: Optional[Sequence] = None
        self._data: Optional[Dict[Hashable, Dict[Hashable, Any]]] = None

        to_process = False
        if not readonly:
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
        # !? This means that the dataset is not dynamic
        if not hasattr(self, "_len"):
            self._len = (
                len(self._indices)
                if self._indices is not None
                else (
                    self.processed_txn.cursor().count()
                    if not self.dynamic
                    else len(self._data)
                )
            )
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
        self.raw_env.close()
        if self.dynamic:
            self.processed_env.close()

    def _check_processed(self, fpath: str):
        return os.path.exists(fpath)

    def _check_processed_complete(self, processed_fpath: str):
        r"""Check if the processed file is complete"""
        env, txn = self.load(processed_fpath, write=False)
        complete = txn.cursor().count() == self.raw_txn.cursor().count()
        env.close()
        return complete

    def _check_has_key(
        self,
        key: Union[Hashable, Tuple[Hashable]],
        _sample: Optional[Dict[Hashable, Any]] = None,
    ) -> bool:
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
            return flag and self._check_has_key(rest, _sample)

    def _assign_txn(
        self, readonly: bool, dynamic: bool, to_process: bool
    ) -> Union[Tuple[lmdb.Environment, lmdb.Transaction], Tuple[None, None]]:
        if readonly:
            return self.raw_env, self.raw_txn
        if not to_process:
            env, txn = self.load(self.processed_fpath, write=False)
            return env, txn
        else:
            self._data = self._process(self.processed_fpath)

        if dynamic:
            return None, None
        else:
            self.save(self.processed_fpath)
            env, txn = self.load(self.processed_fpath, write=False)
            return env, txn

    def _process(self, fpath: str) -> Dict[Hashable, Any]:
        data = {}
        env, txn = self.load(fpath, write=True, replace=True)
        for idx in range(txn.cursor().count()):
            sample = pickle.loads(txn.get(f"{idx}".encode()))
            sample = self.process(sample)
            data[f"{idx}".encode()] = sample
        return data

    def _get_with_keys(
        self,
        key: Union[Hashable, Tuple[Hashable]],
        idx: int,
        _sample: Optional[Dict[Hashable, Any]] = None,
    ):
        r"""Get the data with the given keys"""
        if _sample is None:
            _sample = self.get_fn(idx)

        if not self._check_has_key(key):
            raise KeyError(f"Key {key} not found in the dataset")
        elif not isinstance(key, tuple):
            return _sample[key]
        elif len(key) == 1:
            key = key[0]
            return _sample[key]
        else:
            key, *rest = key
            _sample = _sample[key]
            return self._get_with_keys(rest, idx, _sample)

    def process(self, sample: Dict[Hashable, Any]) -> Dict[Hashable, Any]:
        r"""Users override this method to achieve custom functionality"""
        return sample

    @lru_cache(maxsize=16)
    def _static_get(self, index):
        # TODO: maybe open for transformation plug-ins here or in `__getitem__`
        return self.processed_txn.get(pickle.loads(f"{index}".encode()))

    @lru_cache(maxsize=16)
    def _dynamic_get(self, index):
        return self._data[index]

    @property
    def get_fn(self) -> Any:
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
            readonly=True,
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
            _txn.put(key, pickle.dumps(value, protocol=-1))
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
