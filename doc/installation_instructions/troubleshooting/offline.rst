Airgapped Systems
-----------------

SmartSim assumes that dependencies can be retrieved via the Internet.  The
``smart build`` step can be bypassed by transferring the build artifacts from a
different machine.

.. warning::

    The Redis Source Available License (which licenses RedisAI) prohibits
    distributing binaries to third-parties. Thus, compiled binaries should not
    be shared outside of your organization (see `RSAL v2
    <https://redis.io/legal/rsalv2-agreement/>`_).


The easiest way to accomplish this assumes that you have the following
- A source machine connected to the internet with SmartSim built (referred to as Machine A).
- A target machine not connected to the Internet

.. warning::
    The build and compilation environments of Machine A and B must be compatibile.

**Step 1:** Note the path to SmartSim's ``core`` directory on Machine A

.. code::

    smart info

**Step 2:** tar the ``bin`` and ``lib`` directories

.. code::

    tar -cf smartsim_build_artifacts.tar -C <core-path> bin/ lib/

**Step 3:** Copy the tarball, SmartSim wheel, SmartRedis wheel,
SmartRedis libraries to Machine B (method will vary by machine)

**Step 4:** Install SmartSim and SmartRedis on Machine B

.. code::

    pip install <smartsim-wheel> <smartredis-wheel>

**Step 5:** Find the path to the core directory again with

.. code::

    smart info

**Step 6:** Unpack the tarball to the core directory

.. code::

     tar -xf smartsim_build_artifacts.tar -C <core-path>

**Step 7:** Install the python packages associated with the ML frameworks
(for the default versions reference
``smartsim/_core/_install/configs/mlpackages``)