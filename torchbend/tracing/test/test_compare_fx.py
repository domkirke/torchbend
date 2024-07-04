import os
import traceback
from tabulate import tabulate
import torch
import torchbend as tb
import pytest
from conftest import modules_to_compare


printed_graph_out = os.path.join(os.path.dirname(__file__), "graphs")


def get_tabular_from_graph(self):
        node_specs = [[n.op, n.name, n.target, n.args, n.kwargs]
                      for n in self.nodes]
        return tabulate(node_specs,
              headers=['opcode', 'name', 'target', 'args', 'kwargs'])

@pytest.mark.parametrize("module_config", modules_to_compare)
def test_compare_with_fx(module_config):
    module = module_config.get_module()
    bended_module = tb.BendedModule(module)

    os.makedirs(printed_graph_out, exist_ok=True)
    test_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    # torch.fx graph
    for method in module_config.get_methods():
          args, kwargs = module_config.get_method_args(method)
          with open(os.path.join(printed_graph_out, test_name+"_tfx.txt"), 'w+') as f:
               try:
                    # trace with torch.fx
                    tracer = torch.fx.Tracer()
                    tracer.traced_func_name = method
                    fx_graph = tracer.trace(module)
                    f.write(get_tabular_from_graph(fx_graph))
               except Exception as e:
                    f.write("".join(traceback.format_exception(e)))

          # trace with torchbend
          with open(os.path.join(printed_graph_out, test_name+"_tb.txt"), 'w+') as f:
               try:
                    bended_graph = bended_module.trace(method, **kwargs)
                    bended_module.print_activations(out=os.path.join(printed_graph_out, test_name+"_activations.txt"))
                    f.write(get_tabular_from_graph(bended_graph)) 
               except Exception as e:
                    f.write("".join(traceback.format_exception(e)))



