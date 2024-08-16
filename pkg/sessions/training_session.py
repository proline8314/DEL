import argparse
import copy
import logging
import os
from typing import Any, Dict, Hashable, Optional, Tuple, Union

import networkx as nx
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm

from ..utils.mixin import IArgParse


class TrainingSession(IArgParse):
    def __init__(self):
        pass

    @property
    def _args(self):
        pass

    @property
    def flow_chart(self):
        pass

    def add_args(self, parser):
        pass

    def train_step(self):
        pass

    def valid_step(self):
        pass