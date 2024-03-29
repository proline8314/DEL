import argparse
from abc import ABC, abstractmethod


class IArgParse(ABC):
    @property
    @abstractmethod
    def _args(self):
        raise NotImplementedError

    @abstractmethod
    def add_args(self, parser: argparse.ArgumentParser):
        raise NotImplementedError
    
class IReader(ABC):
    @abstractmethod
    def load(self, fpath: str):
        raise NotImplementedError

class ISaver(ABC):
    @abstractmethod
    def save(self, fpath: str):
        raise NotImplementedError
    
class IFile(IReader, ISaver):
    pass
