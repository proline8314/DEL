import argparse
from abc import ABC, abstractmethod


class IArgParse(ABC):
    @property
    @abstractmethod
    def args(self):
        raise NotImplementedError

    @abstractmethod
    def add_args(self, parser: argparse.ArgumentParser):
        raise NotImplementedError
    
class IFile(ABC):
    @abstractmethod
    def load(self, fpath: str):
        raise NotImplementedError
    
    @abstractmethod
    def save(self, fpath: str):
        raise NotImplementedError
