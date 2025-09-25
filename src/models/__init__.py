from .vanilla import VanillaModel
from .cbm import ConceptBottleneckModel
from .hcbm import HardConceptBottleneckModel
from .cem import ConceptEmbeddingModel
from .mcbm import MinimalConceptBottleneckModel
from .scbm import StochasticConceptBottleneckModel
from .shcbm import StochasticHardConceptBottleneckModel
from .arcbm import AutoregressiveConceptBottleneckModel
from .arhcbm import AutoregressiveHardConceptBottleneckModel
from .baseline import BaselineModel
from .utils import ModelParallel

def get_model(model_type, **kwargs):
    mt = model_type.upper()
    if mt == "VANILLA":
        return VanillaModel(**kwargs)
    elif mt == "CBM":
        return ConceptBottleneckModel(**kwargs)
    elif mt=="HCBM":
        return HardConceptBottleneckModel(**kwargs)
    elif mt == "CEM":
        return ConceptEmbeddingModel(**kwargs)
    elif mt == "MCBM":
        return MinimalConceptBottleneckModel(**kwargs)
    elif mt == "SCBM":
        return StochasticConceptBottleneckModel(**kwargs)
    elif mt == "SHCBM":
        return StochasticHardConceptBottleneckModel(**kwargs)
    elif mt == "ARCBM":
        return AutoregressiveConceptBottleneckModel(**kwargs)
    elif mt == "ARHCBM":
        return AutoregressiveHardConceptBottleneckModel(**kwargs)
    elif mt == "BASELINE":
        return BaselineModel(**kwargs)
    else:
        raise ValueError('model_type must be VANILLA, CBM, CEM, MCBM or SCBM.')