import os
import gin
import torch
from ..module_config import ModuleTestConfig
from ..utils import NotImplementedClass



# try: 
#     import rave
#     config = os.path.join(os.path.dirname(rave.__file__), "configs", "v2.gin")
#     class RAVETest(rave.RAVE):
#         def __new__(cls, *args, **kwargs):
#             gin.parse_config_file(config)
#             return super().__new__(cls, *args, **kwargs)
            
#         def forward(self, x):
#             return super().forward(x)
#     modules_to_compare = [
#         ModuleTestConfig(RAVETest,
#                       (tuple(), dict()),
#                       {'forward': (
#                           tuple(),
#                           {'x': torch.randn(1, 1, 4096)},
#                       )}
#         )
#     ]

# except Exception as e:
#     class RAVETest(NotImplementedClass):
#         pass


__all__ = []
