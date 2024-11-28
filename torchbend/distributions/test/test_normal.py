from random import random
import sys
sys.path.insert(0, ".")
import torch.distributions as dist_torch
from torchbend import distributions as dist
from torchbend.tracing import BendingTracer, Inputs
import torch
import matplotlib.pyplot as plt
import pytest

normal_params = [(torch.zeros(50000), torch.ones(50000), "isotropic"),
                 (torch.full((50000,), 3.), torch.full((50000,), 1.e-3), "narrow_positive"),
                 (torch.full((50000,), -1.), torch.full((50000,), 5.), "wide_negative")]


@pytest.mark.parametrize("mean,var,name", normal_params)
def test_plot(mean, var, name, test_path):
    # name = f"{mean:.4f}_{var:.4f}"
    norm_dist = dist.Normal(mean, var)
    norm_samples = norm_dist.rsample()
    hist_range = [float(norm_samples.min()), float(norm_samples.max())]
    histogram = torch.histogram(norm_samples, 100, range=hist_range, density=True)
    hist_bin = (histogram.bin_edges[1:] + histogram.bin_edges[:-1])/2
    plt.bar(hist_bin, histogram.hist, width=float(hist_bin[1] - hist_bin[0]))
    plt.savefig(f"{test_path}/{name}.pdf")
    plt.close('all')

@pytest.mark.parametrize("mean,var,name", normal_params)
def test_logprob(mean, var, name):
    norm_dist = dist.Normal(mean, var)
    # check log_density
    random_tensor = norm_dist.sample()
    norm_dist_torch = dist_torch.Normal(mean, var)
    assert torch.allclose(norm_dist.log_prob(random_tensor), norm_dist_torch.log_prob(random_tensor))

def test_script(dist_module):
    class NormalModule(dist_module):
        def get_distribution(self, x):
            return dist.Normal(torch.zeros_like(x), torch.ones_like(x))
    module = NormalModule()
    module_scripted = torch.jit.script(module)
    module_graphed = BendingTracer().trace(module, inputs=Inputs(x=torch.randn(1, 100)))