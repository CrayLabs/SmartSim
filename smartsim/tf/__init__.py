from .._core._install.buildenv import Versioner, Version_
from ..error import SmartSimError
from ..log import get_logger

logger = get_logger(__name__)

vers = Versioner()
TF_VERSION = vers.TENSORFLOW

try:
    import tensorflow as tf

    installed_tf = Version_(tf.__version__)
    assert installed_tf >= "2.4.0"

except ImportError:  # pragma: no cover
    raise ModuleNotFoundError(
        f"TensorFlow {TF_VERSION} is not installed. Please install it to use smartsim.tf"
    ) from None
except AssertionError:  # pragma: no cover
    raise SmartSimError(
        f"TensorFlow >= {TF_VERSION} is required for smartsim.tf, you have {tf.__version__}"
    ) from None


from .utils import freeze_model
