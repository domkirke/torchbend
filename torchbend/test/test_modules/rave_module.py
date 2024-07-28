import os
import rave
import gin


config = os.path.join(os.path.dirname(rave.__file__), "configs", "v2.gin")

class RAVETest(rave.RAVE):
    def __new__(cls, *args, **kwargs):
        gin.parse_config_file(config)
        return super().__new__(cls, *args, **kwargs)
        
    def forward(self, x):
        return super().forward(x)

