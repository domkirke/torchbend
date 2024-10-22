from random import random
import sys
sys.path.insert(0, ".")
import torch.distributions as dist_torch
from torchbend import distributions as dist
from torchbend.tracing import BendingTracer, Inputs
import torch
import matplotlib.pyplot as plt
import pytest

bernoulli_params = [(torch.full((50000,), 0.1), "odd"),
                    (torch.full((50000,), 0.5), "central"),
                    (torch.full((50000,), 0.9), "even")]

@pytest.mark.parametrize("probs,name", bernoulli_params)
def test_plot(probs, name, test_path):
    ber_dist = dist.Bernoulli(probs=probs)
    norm_samples = ber_dist.sample()
    histogram = torch.histogram(norm_samples, 2, range=[0., 1.], density=True)
    plt.bar([0., 1.], histogram.hist, width=0.5)
    plt.savefig(f"{test_path}/{name}.pdf")
    plt.close('all')


@pytest.mark.parametrize("probs,name", bernoulli_params)
def test_logprob(probs, name):
    bern_dist = dist.Bernoulli(probs=probs)
    # check log_density
    random_tensor = bern_dist.sample()
    bern_dist_torch = dist_torch.Bernoulli(probs=probs)
    assert torch.allclose(bern_dist.log_prob(random_tensor), bern_dist_torch.log_prob(random_tensor))

def test_script(dist_module):
    class BernoulliProbModule(dist_module):
        def get_distribution(self, x):
            return dist.Bernoulli(probs=torch.rand_like(x))
    class BernoulliLogitsModule(dist_module):
        def get_distribution(self, x):
            return dist.Bernoulli(logits=torch.randn_like(x))
    # module = BernoulliLogitsModule()
    module = BernoulliProbModule()
    module_scripted = torch.jit.script(module)
    module_graphed = BendingTracer().trace(module, inputs=Inputs(x=torch.randn(1, 100)))



# from random import random
# import sys
# sys.path.insert(0, ".")
# import torch.distributions as dist_torch
# from torchbend import distributions as dist
# import torch
# import matplotlib.pyplot as plt
# from conftest import DistModule


# def check_bernoulli_dist_probs(probs, path, name):
#     ber_dist = dist.Bernoulli(probs=probs)
#     norm_samples = ber_dist.sample()
#     histogram = torch.histogram(norm_samples, 2, range=[0., 1.], density=True)
#     plt.bar([0., 1.], histogram.hist, width=0.5)
#     plt.savefig(f"{path}/{name}.pdf")
#     plt.close('all')
#     # check log_density
#     random_tensor = ber_dist.sample()
#     ber_dist_torch = dist_torch.Bernoulli(probs=probs)
#     assert torch.allclose(ber_dist.log_prob(random_tensor), ber_dist_torch.log_prob(random_tensor))

# def check_bernoulli_dist_logits(logits, path, name):

#     # check log_density
#     random_tensor = ber_dist.sample()
#     ber_dist_torch = dist_torch.Bernoulli(logits=logits)
#     assert torch.allclose(ber_dist.log_prob(random_tensor), ber_dist_torch.log_prob(random_tensor))


# def test_bernoulli(test_path):
#     for probs, name in bernoulli_params:
#         check_bernoulli_dist_probs(probs, test_path, "bernoulli_probs_"+name)
#         check_bernoulli_dist_logits(probs_to_logits(probs), test_path, "bernoulli_logits_"+name)
#     module = BernoulliProbModule()
#     module_scripted = torch.jit.script(module)
#     module_scripted(torch.randn(10, 10, 10))
#     module = BernoulliLogitsModule()
#     module_scripted = torch.jit.script(module)
#     module_scripted(torch.randn(10, 10, 10))

