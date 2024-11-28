from random import random
import sys
sys.path.insert(0, ".")
import torch.distributions as dist_torch
from torchbend import distributions as dist
from torchbend.tracing import BendingTracer, Inputs
import torch
import matplotlib.pyplot as plt
import pytest

uniform_params = [(torch.full((50000,), -1.), torch.full((50000,), 1.), "centered")]


@pytest.mark.parametrize("min,max,name", uniform_params)
def test_plot(min, max, name, test_path):
    unif_dist = dist.Uniform(min, max)
    norm_samples = unif_dist.rsample()
    hist_range = [float(min.min()), float(max.max())]
    histogram = torch.histogram(norm_samples, 100, range=hist_range, density=True)
    hist_bin = (histogram.bin_edges[1:] + histogram.bin_edges[:-1])/2
    plt.bar(hist_bin, histogram.hist, width=float(hist_bin[1] - hist_bin[0]))
    plt.savefig(f"{test_path}/{name}.pdf")
    plt.close('all')
    # check log_density
    random_tensor = unif_dist.sample()
    unif_dist_torch = dist_torch.Uniform(min, max)
    assert torch.allclose(unif_dist.log_prob(random_tensor), unif_dist_torch.log_prob(random_tensor))


@pytest.mark.parametrize("mean,var,name", uniform_params)
def test_logprob(mean, var, name):
    norm_dist = dist.Normal(mean, var)
    # check log_density
    random_tensor = norm_dist.sample()
    norm_dist_torch = dist_torch.Normal(mean, var)
    assert torch.allclose(norm_dist.log_prob(random_tensor), norm_dist_torch.log_prob(random_tensor))

def test_script(dist_module):
    class UniformModule(dist_module):
        def get_distribution(self, x):
            return dist.Uniform(torch.full_like(x, -1), torch.full_like(x, 1))
    module = UniformModule()
    module_scripted = torch.jit.script(module)
    module_graphed = BendingTracer().trace(module, inputs=Inputs(x=torch.randn(1, 100)))

