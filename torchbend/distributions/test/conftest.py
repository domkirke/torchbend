import os, pytest
import torch, torch.nn as nn


@pytest.fixture
def test_path():
    figure_path = f"{os.path.dirname(os.path.abspath(__file__))}/figures"
    if not os.path.isdir(figure_path):
        os.makedirs(figure_path)
    return figure_path

@pytest.fixture
def closeall_tolerance():
    return 1e-4


class DistModule(nn.Module):
    def __init__(self):
        super().__init__()
    def get_distribution(self, x):
        return NotImplemented
    def forward(self, x: torch.Tensor):
        dist = self.get_distribution(x)
        return dist

@pytest.fixture
def dist_module():
    return DistModule
