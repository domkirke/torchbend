"""A module that marks with a description and metadata, for scripting (a file,
not a test body: TorchScript reads the source)."""
import torch
import torch.nn as nn

from torchbend import mark


class ScriptableMarked(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.a = nn.Linear(4, 4)

    def forward(self, x):
        return mark(self.a(x), name="z", description="the projection",
                    meta={"step": 1, "unit": "features"})
