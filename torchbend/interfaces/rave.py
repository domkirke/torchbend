import rave as ravelib
from .base import Interface
import gin

class BendingRAVEException(Exception):
    pass


class BendedRAVE(Interface):
    _imported_callbacks_ = []

    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    @staticmethod
    def load_model(model_path):
        config_path = ravelib.core.search_for_config(model_path)
        if config_path is None:
            raise BendingRAVEException('config not found in folder %s'%model_path)
        gin.parse_config_file(config_path)
        model = ravelib.RAVE()
        run = ravelib.core.search_for_run(model_path)
        if run is None:
            raise BendingRAVEException("run not found in folder %s"%model_path)
        model = model.load_from_checkpoint(run)
        return model

    def _bend_model(self, model):
        self._model = model


__all__ = ['BendedRAVE']