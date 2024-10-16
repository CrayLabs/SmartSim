Custom ML backends
------------------

The ML backends (Torch, ONNX Runtime, and Tensorflow) and their associated
python packages have different versions and indices that can be supported based
on the intended device (CPU, ROCM, CUDA-11, or CUDA-12). SmartSim stores this
information in JSON files within the ``smartsim/_core/_install/configs/mlpackages``
directory. If a different version or variant is needed, these can be specified
using ``smart build --config-dir <path-to-config-dir>``. The following is the
JSON file used for Linux with CUDA-12.

.. literalinclude:: ../../../smartsim/_core/_install/configs/mlpackages/Linux64CUDA12.json

The following table explains what each of the main fields are:

.. list-table:: MLPackages fields
    :widths: 25 50
    :header-rows: 1

    * - Field Name
      - Description
    * - name
      - The name of the C++ frontend to the ML package itself (e.g. libtorch)
    * - version
      - A string used to identify the version of the library. Note that this does not have
        an effect on the build process itself, but is used to display information
    * - pip_index
      - The pip index from which to install the python packages associated with this ML package
    * - lib_source
      - The location of the archive which contains the ML backend. If this is a URL, the file
        will be downloaded, otherwise if this is a local path, the archive will be copied to
        the build library and extracted
    * - rai_patches
      - Patch RedisAI source code with modifications needed by this ML package