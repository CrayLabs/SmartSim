from .._core._install.buildenv import Versioner
from ..error import SmartSimError
from ..log import get_logger

logger = get_logger(__name__)

vers = Versioner()
TF_VERSION = vers.TENSORFLOW

try:
    import tensorflow as tf

    tf_version = tf.__version__.split(".")
    assert int(tf_version[0]) == 2 and int(tf_version[1]) >= 4

except ImportError:  # pragma: no cover
    raise SmartSimError(
        f"TensorFlow {TF_VERSION} is not installed. Please install it to use smartsim.tf"
    ) from None
except AssertionError:  # pragma: no cover
    raise SmartSimError(
        f"TensorFlow {TF_VERSION} is required for smartsim.tf, you have {tf_version}"
    ) from None


from .utils import freeze_model
