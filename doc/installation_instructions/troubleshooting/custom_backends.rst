Custom ML backends
------------------

The ML backends (Torch, ONNX Runtime, and Tensorflow) and their associated
python packages have different versions and indices that can be supported based
on the intended device (CPU, ROCm, CUDA-11, or CUDA-12). The officially
supported backends are stored in JSON files within the
``smartsim/_core/_install/configs/mlpackages`` directory.

To customize the version of a backend and/or package, we recommend that you use
a configuration shipped with SmartSim as a template (for example the one at the
end of this section). Copy the file and update as needed. Afterwards, use
``smart build --config-dir`` to tell the build process to use custom
configuration(s).

The following table describes the main fields needed to define a machine learning
backend used by RedisAI.

.. list-table:: MLPackages fields
    :widths: 15 60
    :header-rows: 1

    * - Field Name
      - Description
    * - ``name``
      - The name of the C++ frontend to the ML package itself (e.g. libtorch)
    * - ``version``
      - A string used to identify the version of the library. Note that this does not have
        an effect on the build process itself, but is used to display information
    * - ``pip_index``
      - The pip index from which to install the python packages associated with this ML package
    * - ``lib_source``
      - The location of the archive which contains the ML backend. If this is a URL, the file
        will be downloaded, otherwise if this is a local path, the archive will be copied to
        the build library and extracted
    * - ``rai_patches``
      - Patch RedisAI source code with modifications needed by this ML package

As an example, the following file describes the ML frameworks for Linux on CUDA-12 devices:

.. literalinclude:: ../../../smartsim/_core/_install/configs/mlpackages/LinuxX64CUDA12.json
