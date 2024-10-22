# from random import random
# import sys
# sys.path.insert(0, ".")
# import torch.distributions as dist_torch
# from torchbend import distributions as dist
# import torch
# import matplotlib.pyplot as plt
# from .conftest import DistModule## Testing bernoulli distribution


# class CategoricalProbsModule(DistModule):
#     def get_distribution(self, x):
#         return dist.Bernoulli(probs=torch.rand_like(x))

# # class CategoricalLogitsModule(DistModule):
# #     def get_distribution(self, x):
# #         return dist.Bernoulli(logits=torch.randn_like(x))

# def check_categorical_dist_probs(probs, path, name):
#     cat_dist = dist.Categorical(probs=probs)
#     cat_samples = cat_dist.sample()
#     classes = list(range(probs.size(-1)))
#     count = [(cat_samples == x).nonzero().numel() for x in classes]
#     plt.bar(classes, count, width=1)
#     plt.savefig(f"{path}/{name}.pdf")
#     plt.close('all')
#     # check log_density
#     random_tensor = cat_dist.sample()
#     cat_dist_torch = dist_torch.Categorical(probs=probs)
#     assert torch.allclose(cat_dist.log_prob(random_tensor), cat_dist_torch.log_prob(random_tensor))

# def check_categorical_dist_logits(probs, path, name):
#     cat_dist = dist.Categorical(logits=probs)
#     cat_samples = cat_dist.sample()
#     classes = list(range(probs.size(-1)))
#     count = [(cat_samples == x).nonzero().numel() for x in classes]
#     plt.bar(classes, count, width=1)
#     plt.savefig(f"{path}/{name}.pdf")
#     plt.close('all')
#     # check log_density
#     random_tensor = cat_dist.sample()
#     cat_dist_torch = dist_torch.Categorical(logits=probs)
#     assert torch.allclose(cat_dist.log_prob(random_tensor), cat_dist_torch.log_prob(random_tensor))

# categorical_params= [(torch.cat([torch.full((50000,5), 0.1), torch.full((50000,1), 1.0), torch.full((50000,4), 0.1)], -1) , "class_six"),
#                      (torch.cat([torch.full((50000,1), 0.1), torch.full((50000,1), 1.0), torch.full((50000,8), 0.1)], -1) , "class_two"),
#                      (torch.full((50000,10), 0.1), "uniform")]

# def test_categorical(test_path):
#     for probs, name in categorical_params:
#         check_categorical_dist_probs(probs, test_path, "categorical_probs_"+name)
#         check_categorical_dist_logits(probs_to_logits(probs), test_path, "categorical_logits_"+name)
#     module = BernoulliProbModule()
#     module_scripted = torch.jit.script(module)
#     module_scripted(torch.randn(10, 10, 10))
#     module = BernoulliLogitsModule()
#     module_scripted = torch.jit.script(module)
#     module_scripted(torch.randn(10, 10, 10))
