import os
from pathlib import Path

"""
Verify home directory only contains a single directory.
This script is intended to be run by a container test with a test directory
mounted into the $HOME directory.
"""

directories = os.listdir(str(Path.home()))
print(directories)
assert len(directories) == 1
