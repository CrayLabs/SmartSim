Site Installation
=================

Certain HPE customer machines have a site installation of SmartSim. This means
that users can bypass the ``smart build`` step that builds the ML backends and
the Redis binaries. Users on these platforms can install SmartSim from PyPI or
from source with the following steps replacing ``COMPILER_VERSION`` and
``SMARTSIM_VERSION`` with the desired entries.

.. code:: bash

    module use -a /lus/scratch/smartsim/local/modulefiles
    module load cudatoolkit/11.8 cudnn smartsim-deps/COMPILER_VERSION/SMARTSIM_VERSION
    pip install smartsim
    smart build --skip-backends --device gpu [--onnx]
