import torch, torch.nn as nn

class VideoTransform(nn.Module):
    invertible = True
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x

    def invert(self, x: torch.Tensor):
        return x
