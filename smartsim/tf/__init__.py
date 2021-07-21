from ..error import SmartSimError
from ..constants import TF_VERSION

try:
    import tensorflow as tf
    assert tf.__version__ == TF_VERSION

except (ImportError, AssertionError):
    raise SmartSimError(
        f"Tensorflow {TF_VERSION} is not installed. Please install it to use smartsim.tf") from None


from .utils import freeze_model