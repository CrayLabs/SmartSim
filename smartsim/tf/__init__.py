import sys

from ..constants import TF_VERSION
from ..error import SmartSimError
from ..utils.log import get_logger

logger = get_logger(__name__)

try:
    import tensorflow as tf

    # in python >= 3.9 Tensorflow only supports >= tf 2.5
    if sys.version_info.minor >= 9:
        logger.warning(
            f"TensorFlow >= 2.5 is not offcially supported, SmartSim uses TensorFlow {TF_VERSION}"
        )

    tf_version = tf.__version__.split(".")
    assert int(tf_version[0]) == 2 and int(tf_version[1]) >= 4

except ImportError:
    raise SmartSimError(
        f"TensorFlow {TF_VERSION} is not installed. Please install it to use smartsim.tf"
    ) from None
except AssertionError:
    raise SmartSimError(
        f"TensorFlow {TF_VERSION} is required for smartsim.tf, you have {tf_version}"
    ) from None


from .utils import freeze_model
