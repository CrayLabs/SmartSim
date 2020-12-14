__author__ = "Sam Partee"


try:
    from smartsim import *

    _top_import_error = None
except Exception as e:
    _top_import_error = e


def test_import_ss():
    # Test either above import has failed for some reason
    # "import *" is discouraged outside of the module level, hence we
    # rely on setting up the variable above
    assert _top_import_error is None


test_import_ss()
