*********
Changelog
*********

Listed here are the changes between each release of SmartSim
and SmartRedis.

Jump to :ref:`SmartRedis Changelog <changelog>`


SmartSim
========


0.5.1
-----

Released on 14 September, 2023

Description

- Add typehints throughout the SmartSim codebase
- Provide support for Slurm heterogeneous jobs
- Provide better support for `PalsMpiexecSettings`
- Allow for easier inspection of SmartSim entities
- Log ignored error messages from `sacct`
- Fix colocated db preparation bug when using `JsrunSettings`
- Fix bug when user specify CPU and devices greater than 1
- Fix bug when get_allocation called with reserved keywords
- Enabled mypy in CI for better type safety
- Mitigate additional suppressed pylint errors
- Update linting support and apply to existing errors
- Various improvements to the `smart` CLI
- Various documentation improvements
- Various test suite improvements

Detailed Notes

- Add methods to allow users to inspect files attached to models and ensembles. (PR352_)
- Add a `smart info` target to provide rudimentary information about the SmartSim installation. (PR350_)
- Remove unnecessary generation producing unexpected directories in the test suite. (PR349_)
- Add support for heterogeneous jobs to `SrunSettings` by allowing users to set the `--het-group` parameter. (PR346_)
- Provide clearer guidelines on how to contribute to SmartSim. (PR344_)
- Integrate `PalsMpiexecSettings` into the `Experiment` factory methods when using the `"pals"` launcher. (PR343_)
- Create public properties where appropriate to mitigate `protected-access` errors. (PR341_)
- Fix a failure to execute `_prep_colocated_db` due to incorrect named attr check. (PR339_)
- Enabled and mitigated mypy `disallow_any_generics` and `warn_return_any`. (PR338_)
- Add a `smart validate` target to provide a simple smoke test to assess a SmartSim build. (PR336_, PR351_)
- Add typehints to `smartsim._core.launcher.step.*`. (PR334_)
- Log errors reported from slurm WLM when attempts to retrieve status fail. (PR331_, PR332_)
- Fix incorrectly formatted positional arguments in log format strings. (PR330_)
- Ensure that launchers pass environment variables to unmanaged job steps. (PR329_)
- Add additional tests surrounding the `RAI_PATH` configuration environment variable. (PR328_)
- Remove unnecessary execution of unescaped shell commands. (PR327_)
- Add error if user calls get_allocation with reserved keywords in slurm get_allocation. (PR325_)
- Add error when user requests CPU with devices greater than 1 within add_ml_model and add_script. (PR324_)
- Update documentation surrounding ensemble key prefixing. (PR322_)
- Fix formatting of the Frontier site installation. (PR321_)
- Update pylint dependency, update .pylintrc, mitigate non-breaking issues, suppress api breaks. (PR311_)
- Refactor the `smart` CLI to use subparsers for better documentation and extension. (PR308_)

.. _PR352: https://github.com/CrayLabs/SmartSim/pull/352
.. _PR351: https://github.com/CrayLabs/SmartSim/pull/351
.. _PR350: https://github.com/CrayLabs/SmartSim/pull/350
.. _PR349: https://github.com/CrayLabs/SmartSim/pull/349
.. _PR346: https://github.com/CrayLabs/SmartSim/pull/346
.. _PR344: https://github.com/CrayLabs/SmartSim/pull/344
.. _PR343: https://github.com/CrayLabs/SmartSim/pull/343
.. _PR341: https://github.com/CrayLabs/SmartSim/pull/341
.. _PR339: https://github.com/CrayLabs/SmartSim/pull/339
.. _PR338: https://github.com/CrayLabs/SmartSim/pull/338
.. _PR336: https://github.com/CrayLabs/SmartSim/pull/336
.. _PR334: https://github.com/CrayLabs/SmartSim/pull/334
.. _PR332: https://github.com/CrayLabs/SmartSim/pull/332
.. _PR331: https://github.com/CrayLabs/SmartSim/pull/331
.. _PR330: https://github.com/CrayLabs/SmartSim/pull/330
.. _PR329: https://github.com/CrayLabs/SmartSim/pull/329
.. _PR328: https://github.com/CrayLabs/SmartSim/pull/328
.. _PR327: https://github.com/CrayLabs/SmartSim/pull/327
.. _PR325: https://github.com/CrayLabs/SmartSim/pull/325
.. _PR324: https://github.com/CrayLabs/SmartSim/pull/324
.. _PR322: https://github.com/CrayLabs/SmartSim/pull/322
.. _PR321: https://github.com/CrayLabs/SmartSim/pull/321
.. _PR311: https://github.com/CrayLabs/SmartSim/pull/311
.. _PR308: https://github.com/CrayLabs/SmartSim/pull/308


0.5.0
-----

Released on 6 July, 2023

Description

A full list of changes and detailed notes can be found below:

- Update SmartRedis dependency to v0.4.1
- Fix tests for db models and scripts
- Fix add_ml_model() and add_script() documentation, tests, and code
- Remove `requirements.txt` and other places where dependencies were defined
- Replace `limit_app_cpus` with `limit_db_cpus` for co-located orchestrators
- Remove wait time associated with Experiment launch summary
- Update and rename Redis conf file
- Migrate from redis-py-cluster to redis-py
- Update full test suite to not require a TF wheel at test time
- Update doc strings
- Remove deprecated code
- Relax the coloredlogs version
- Update Fortran tutorials for SmartRedis
- Add support for multiple network interface binding in Orchestrator and Colocated DBs
- Add typehints and static analysis

Detailed notes

- Updates SmartRedis to the most current release (PR316_)
- Fixes and enhancements to documentation (PR317_, PR314_, PR287_)
- Various fixes and enhancements to the test suite (PR315_, PR312_, PR310_, PR302_, PR283_)
- Fix a defect in the tests related to database models and scripts that was
  causing key collisions when testing on workload managers (PR313_)
- Remove `requirements.txt` and other places where dependencies were defined. (PR307_)
- Fix defect where dictionaries used to create run settings can be changed
  unexpectedly due to copy-by-ref (PR305_)
- The underlying code for Model.add_ml_model() and Model.add_script() was fixed
  to correctly handle multi-GPU configurations.  Tests were updated to run on
  non-local launchers.  Documentation was updated and fixed.  Also, the default
  testing interface has been changed to lo instead of ipogif. (PR304_)
- Typehints have been added. A makefile target `make check-mypy` executes static
  analysis with mypy. (PR295_, PR301_, PR303_)
- Replace `limit_app_cpus` with `limit_db_cpus` for co-located orchestrators.
  This resolves some incorrect behavior/assumptions about how the application
  would be pinned.  Instead, users should directly specify the binding options in
  their application using the options appropriate for their launcher (PR306_)
- Simplify code in `random_permutations` parameter generation strategy (PR300_)
- Remove wait time associated with Experiment launch summary (PR298_)
- Update Redis conf file to conform with Redis v7.0.5 conf file (PR293_)
- Migrate from redis-py-cluster to redis-py for cluster status checks (PR292_)
- Update full test suite to no longer require a tensorflow wheel to be available at test time. (PR291_)
- Correct spelling of colocated in doc strings (PR290_)
- Deprecated launcher-specific orchestrators, constants, and ML
  utilities were removed. (PR289_)
- Relax the coloredlogs version to be greater than 10.0 (PR288_)
- Update the Github Actions runner image from `macos-10.15`` to `macos-12``. The
  former began deprecation in May 2022 and was finally removed in May 2023. (PR285_)
- The Fortran tutorials had not been fully updated to show how to handle
  return/error codes. These have now all been updated. (PR284_)
- Orchestrator and Colocated DB now accept a list of interfaces to bind to. The
  argument name is still `interface` for backward compatibility reasons. (PR281_)
- Typehints have been added to public APIs. A makefile target to execute static
  analysis with mypy is available `make check-mypy`. (PR295_)

.. _PR317: https://github.com/CrayLabs/SmartSim/pull/317
.. _PR316: https://github.com/CrayLabs/SmartSim/pull/316
.. _PR315: https://github.com/CrayLabs/SmartSim/pull/314
.. _PR314: https://github.com/CrayLabs/SmartSim/pull/314
.. _PR313: https://github.com/CrayLabs/SmartSim/pull/313
.. _PR312: https://github.com/CrayLabs/SmartSim/pull/312
.. _PR310: https://github.com/CrayLabs/SmartSim/pull/310
.. _PR307: https://github.com/CrayLabs/SmartSim/pull/307
.. _PR306: https://github.com/CrayLabs/SmartSim/pull/306
.. _PR305: https://github.com/CrayLabs/SmartSim/pull/305
.. _PR304: https://github.com/CrayLabs/SmartSim/pull/304
.. _PR303: https://github.com/CrayLabs/SmartSim/pull/303
.. _PR302: https://github.com/CrayLabs/SmartSim/pull/302
.. _PR301: https://github.com/CrayLabs/SmartSim/pull/301
.. _PR300: https://github.com/CrayLabs/SmartSim/pull/300
.. _PR298: https://github.com/CrayLabs/SmartSim/pull/298
.. _PR295: https://github.com/CrayLabs/SmartSim/pull/295
.. _PR293: https://github.com/CrayLabs/SmartSim/pull/293
.. _PR292: https://github.com/CrayLabs/SmartSim/pull/292
.. _PR291: https://github.com/CrayLabs/SmartSim/pull/291
.. _PR290: https://github.com/CrayLabs/SmartSim/pull/290
.. _PR289: https://github.com/CrayLabs/SmartSim/pull/289
.. _PR288: https://github.com/CrayLabs/SmartSim/pull/288
.. _PR287: https://github.com/CrayLabs/SmartSim/pull/287
.. _PR285: https://github.com/CrayLabs/SmartSim/pull/285
.. _PR284: https://github.com/CrayLabs/SmartSim/pull/284
.. _PR283: https://github.com/CrayLabs/SmartSim/pull/283
.. _PR281: https://github.com/CrayLabs/SmartSim/pull/281

0.4.2
-----

Released on April 12, 2023

Description

This release of SmartSim had a focus on polishing and extending exiting
features already provided by SmartSim. Most notably, this release provides
support to allow users to colocate their models with an orchestrator using
Unix domain sockets and support for launching models as batch jobs.

Additionally, SmartSim has updated its tool chains to provide a better user
experience. Notably, SmarSim can now be used with Python 3.10, Redis 7.0.5, and
RedisAI 1.2.7. Furthermore, SmartSim now utilizes SmartRedis's aggregation lists to
streamline the use and extension of ML data loaders, making working with popular
machine learning frameworks in SmartSim a breeze.

A full list of changes and detailed notes can be found below:

- Add support for colocating an orchestrator over UDS
- Add support for Python 3.10, deprecate support for Python 3.7 and RedisAI 1.2.3
- Drop support for Ray
- Update ML data loaders to make use of SmartRedis's aggregation lists
- Allow for models to be launched independently as batch jobs
- Update to current version of Redis to 7.0.5
- Add support for RedisAI 1.2.7, pyTorch 1.11.0, Tensorflow 2.8.0, ONNXRuntime 1.11.1
- Fix bug in colocated database entrypoint when loading PyTorch models
- Fix test suite behavior with environment variables

Detailed Notes

- Running some tests could result in some SmartSim-specific environment variables to be set. Such environment variables are now reset
  after each test execution. Also, a warning for environment variable usage in Slurm was added, to make the user aware in case an environment
  variable will not be assigned the desired value with `--export`. (PR270_)
- The PyTorch and TensorFlow data loaders were update to make use of aggregation lists. This breaks their API, but makes them easier to use. (PR264_)
- The support for Ray was dropped, as its most recent versions caused problems when deployed through SmartSim.
  We plan to release a separate add-on library to accomplish the same results. If
  you are interested in getting the Ray launch functionality back in your workflow, please get in touch with us! (PR263_)
- Update from Redis version 6.0.8 to 7.0.5. (PR258_)
- Adds support for Python 3.10 without the ONNX machine learning backend. Deprecates support for
  Python 3.7 as it will stop receiving security updates. Deprecates support for RedisAI 1.2.3.
  Update the build process to be able to correctly fetch supported dependencies. If a user
  attempts to build an unsupported dependency, an error message is shown highlighting the
  discrepancy. (PR256_)
- Models were given a `batch_settings` attribute. When launching a model through `Experiment.start`
  the `Experiment` will first check for a non-nullish value at that attribute. If the check is
  satisfied, the `Experiment` will attempt to wrap the underlying run command in a batch job using
  the object referenced at `Model.batch_settings` as the batch settings for the job. If the check
  is not satisfied, the `Model` is launched in the traditional manner as a job step. (PR245_)
- Fix bug in colocated database entrypoint stemming from uninitialized variables. This bug affects PyTorch models being loaded into the database. (PR237_)
- The release of RedisAI 1.2.7 allows us to update support for recent versions of PyTorch, Tensorflow, and ONNX (PR234_)
- Make installation of correct Torch backend more reliable according to instruction from PyTorch
- In addition to TCP, add UDS support for colocating an orchestrator with models. Methods
  `Model.colocate_db_tcp` and `Model.colocate_db_uds` were added to expose this functionality.
  The `Model.colocate_db` method remains and uses TCP for backward compatibility (PR246_)

.. _PR270: https://github.com/CrayLabs/SmartSim/pull/270
.. _PR264: https://github.com/CrayLabs/SmartSim/pull/264
.. _PR263: https://github.com/CrayLabs/SmartSim/pull/263
.. _PR258: https://github.com/CrayLabs/SmartSim/pull/258
.. _PR256: https://github.com/CrayLabs/SmartSim/pull/256
.. _PR246: https://github.com/CrayLabs/SmartSim/pull/246
.. _PR245: https://github.com/CrayLabs/SmartSim/pull/245
.. _PR237: https://github.com/CrayLabs/SmartSim/pull/237
.. _PR234: https://github.com/CrayLabs/SmartSim/pull/234


0.4.1
-----

Released on June 24, 2022

Description:
This release of SmartSim introduces a new experimental feature to help make
SmartSim workflows more portable: the ability to run simulations models in a
container via Singularity. This feature has been tested on a small number of
platforms and we encourage users to provide feedback on its use.

We have also made improvements in a variety of areas: new utilities to load
scripts and machine learning models into the database directly from SmartSim
driver scripts and install-time choice to use either `KeyDB` or `Redis` for the
Orchestrator. The `RunSettings` API is now more consistent across subclasses. Another
key focus of this release was to aid new SmartSim users by including more
extensive tutorials and improving the documentation. The docker image containing
the SmartSim tutorials now also includes a tutorial on online training.


Launcher improvements

    - New methods for specifying `RunSettings` parameters (SmartSim-PR166_) (SmartSim-PR170_)
    - Better support for `mpirun`, `mpiexec`, and `orterun` as launchers (SmartSim-PR186_)
    - Experimental: add support for running models via Singularity (SmartSim-PR204_)

Documentation and tutorials

    - Tutorial updates (SmartSim-PR155_) (SmartSim-PR203_) (SmartSim-PR208_)
    - Add SmartSim Zoo info to documentation (SmartSim-PR175_)
    - New tutorial for demonstrating online training (SmartSim-PR176_) (SmartSim-PR188_)

General improvements and bug fixes

    - Set models and scripts at the driver level (SmartSim-PR185_)
    - Optionally use KeyDB for the orchestrator (SmartSim-PR180_)
    - Ability to specify system-level libraries (SmartSim-PR154_) (SmartSim-PR182_)
    - Fix the handling of LSF gpus_per_shard (SmartSim-PR164_)
    - Fix error when re-running `smart build` (SmartSim-PR165_)
    - Fix generator hanging when tagged configuration variables are missing (SmartSim-PR177_)

Dependency updates

    - CMake version from 3.10 to 3.13 (SmartSim-PR152_)
    - Update click to 8.0.2 (SmartSim-PR200_)

.. _SmartSim-PR152: https://github.com/CrayLabs/SmartSim/pull/152
.. _SmartSim-PR154: https://github.com/CrayLabs/SmartSim/pull/154
.. _SmartSim-PR155: https://github.com/CrayLabs/SmartSim/pull/155
.. _SmartSim-PR164: https://github.com/CrayLabs/SmartSim/pull/164
.. _SmartSim-PR165: https://github.com/CrayLabs/SmartSim/pull/165
.. _SmartSim-PR166: https://github.com/CrayLabs/SmartSim/pull/166
.. _SmartSim-PR170: https://github.com/CrayLabs/SmartSim/pull/170
.. _SmartSim-PR175: https://github.com/CrayLabs/SmartSim/pull/175
.. _SmartSim-PR176: https://github.com/CrayLabs/SmartSim/pull/176
.. _SmartSim-PR177: https://github.com/CrayLabs/SmartSim/pull/177
.. _SmartSim-PR180: https://github.com/CrayLabs/SmartSim/pull/180
.. _SmartSim-PR182: https://github.com/CrayLabs/SmartSim/pull/182
.. _SmartSim-PR185: https://github.com/CrayLabs/SmartSim/pull/185
.. _SmartSim-PR186: https://github.com/CrayLabs/SmartSim/pull/186
.. _SmartSim-PR188: https://github.com/CrayLabs/SmartSim/pull/188
.. _SmartSim-PR200: https://github.com/CrayLabs/SmartSim/pull/200
.. _SmartSim-PR203: https://github.com/CrayLabs/SmartSim/pull/203
.. _SmartSim-PR204: https://github.com/CrayLabs/SmartSim/pull/204
.. _SmartSim-PR208: https://github.com/CrayLabs/SmartSim/pull/208

0.4.0
-----

Released on Feb 11, 2022

Description:
In this release SmartSim continues to promote ease of use.
To this end SmartSim has introduced new portability features
that allow users to abstract away their targeted hardware,
while providing even more compatibility with existing
libraries.

A new feature, Co-located orchestrator deployments has
been added which provides scalable online inference
capabilities that overcome previous performance limitations
in seperated orchestrator/application deployments.
For more information on advantages of co-located deployments,
see the Orchestrator section of the SmartSim documentation.

The SmartSim build was significantly improved to increase
customization of build toolchain and the ``smart`` command
line inferface was expanded.

Additional tweaks and upgrades have also been
made to ensure an optimal experience. Here is a
comprehensive list of changes made in SmartSim 0.4.0.


Orchestrator Enhancements:

 - Add Orchestrator Co-location (SmartSim-PR139_)
 - Add Orchestrator configuration file edit methods (SmartSim-PR109_)

Emphasize Driver Script Portability:

 - Add ability to create run settings through an experiment (SmartSim-PR110_)
 - Add ability to create batch settings through an experiment (SmartSim-PR112_)
 - Add automatic launcher detection to experiment portability functions (SmartSim-PR120_)

Expand Machine Learning Library Support:

 - Data loaders for online training in Keras/TF and Pytorch (SmartSim-PR115_) (SmartSim-PR140_)
 - ML backend versions updated with expanded support for multiple versions (SmartSim-PR122_)
 - Launch Ray internally using ``RunSettings`` (SmartSim-PR118_)
 - Add Ray cluster setup and deployment to SmartSim (SmartSim-PR50_)

Expand Launcher Setting Options:

 - Add ability to use base ``RunSettings`` on a Slurm, PBS, or Cobalt launchers (SmartSim-PR90_)
 - Add ability to use base ``RunSettings`` on LFS launcher (SmartSim-PR108_)

Deprecations and Breaking Changes

 - Orchestrator classes combined into single implementation for portability (SmartSim-PR139_)
 - ``smartsim.constants`` changed to ``smartsim.status`` (SmartSim-PR122_)
 - ``smartsim.tf`` migrated to ``smartsim.ml.tf`` (SmartSim-PR115_) (SmartSim-PR140_)
 - TOML configuration option removed in favor of environment variable approach (SmartSim-PR122_)

General Improvements and Bug Fixes:

 - Improve and extend parameter handling (SmartSim-PR107_) (SmartSim-PR119_)
 - Abstract away non-user facing implementation details (SmartSim-PR122_)
 - Add various dimensions to the CI build matrix for SmartSim testing (SmartSim-PR130_)
 - Add missing functions to LSFSettings API (SmartSim-PR113_)
 - Add RedisAI checker for installed backends (SmartSim-PR137_)
 - Remove heavy and unnecessary dependencies (SmartSim-PR116_) (SmartSim-PR132_)
 - Fix LSFLauncher and LSFOrchestrator (SmartSim-PR86_)
 - Fix over greedy Workload Manager Parsers (SmartSim-PR95_)
 - Fix Slurm handling of comma-separated env vars (SmartSim-PR104_)
 - Fix internal method calls (SmartSim-PR138_)

Documentation Updates:

 - Updates to documentation build process (SmartSim-PR133_) (SmartSim-PR143_)
 - Updates to documentation content (SmartSim-PR96_) (SmartSim-PR129_) (SmartSim-PR136_) (SmartSim-PR141_)
 - Update SmartSim Examples (SmartSim-PR68_) (SmartSim-PR100_)


.. _SmartSim-PR50: https://github.com/CrayLabs/SmartSim/pull/50
.. _SmartSim-PR68: https://github.com/CrayLabs/SmartSim/pull/68
.. _SmartSim-PR86: https://github.com/CrayLabs/SmartSim/pull/86
.. _SmartSim-PR90: https://github.com/CrayLabs/SmartSim/pull/90
.. _SmartSim-PR95: https://github.com/CrayLabs/SmartSim/pull/95
.. _SmartSim-PR96: https://github.com/CrayLabs/SmartSim/pull/96
.. _SmartSim-PR100: https://github.com/CrayLabs/SmartSim/pull/100
.. _SmartSim-PR104: https://github.com/CrayLabs/SmartSim/pull/104
.. _SmartSim-PR107: https://github.com/CrayLabs/SmartSim/pull/107
.. _SmartSim-PR108: https://github.com/CrayLabs/SmartSim/pull/108
.. _SmartSim-PR109: https://github.com/CrayLabs/SmartSim/pull/109
.. _SmartSim-PR110: https://github.com/CrayLabs/SmartSim/pull/110
.. _SmartSim-PR112: https://github.com/CrayLabs/SmartSim/pull/112
.. _SmartSim-PR113: https://github.com/CrayLabs/SmartSim/pull/113
.. _SmartSim-PR115: https://github.com/CrayLabs/SmartSim/pull/115
.. _SmartSim-PR116: https://github.com/CrayLabs/SmartSim/pull/116
.. _SmartSim-PR118: https://github.com/CrayLabs/SmartSim/pull/118
.. _SmartSim-PR119: https://github.com/CrayLabs/SmartSim/pull/119
.. _SmartSim-PR120: https://github.com/CrayLabs/SmartSim/pull/120
.. _SmartSim-PR122: https://github.com/CrayLabs/SmartSim/pull/122
.. _SmartSim-PR129: https://github.com/CrayLabs/SmartSim/pull/129
.. _SmartSim-PR130: https://github.com/CrayLabs/SmartSim/pull/130
.. _SmartSim-PR132: https://github.com/CrayLabs/SmartSim/pull/132
.. _SmartSim-PR133: https://github.com/CrayLabs/SmartSim/pull/133
.. _SmartSim-PR136: https://github.com/CrayLabs/SmartSim/pull/136
.. _SmartSim-PR137: https://github.com/CrayLabs/SmartSim/pull/137
.. _SmartSim-PR138: https://github.com/CrayLabs/SmartSim/pull/138
.. _SmartSim-PR139: https://github.com/CrayLabs/SmartSim/pull/139
.. _SmartSim-PR140: https://github.com/CrayLabs/SmartSim/pull/140
.. _SmartSim-PR141: https://github.com/CrayLabs/SmartSim/pull/141
.. _SmartSim-PR143: https://github.com/CrayLabs/SmartSim/pull/143


0.3.2
-----

Released on August 10, 2021

Description:

 - Upgraded RedisAI backend to 1.2.3 (SmartSim-PR69_)
 - PyTorch 1.7.1, TF 2.4.2, and ONNX 1.6-7 (SmartSim-PR69_)
 - LSF launcher for IBM machines (SmartSim-PR62_)
 - Improved code coverage by adding more unit tests (SmartSim-PR53_)
 - Orchestrator methods to get address and check status (SmartSim-PR60_)
 - Added Manifest object that tracks deployables in Experiments (SmartSim-PR61_)
 - Bug fixes (SmartSim-PR52_) (SmartSim-PR58_) (SmartSim-PR67_) (SmartSim-PR73_)
 - Updated documentation and examples (SmartSim-PR51_) (SmartSim-PR57_) (SmartSim-PR71_)
 - Improved IP address aquisition (SmartSim-PR72_)
 - Binding database to network interfaces

.. _SmartSim-PR51: https://github.com/CrayLabs/SmartSim/pull/51
.. _SmartSim-PR52: https://github.com/CrayLabs/SmartSim/pull/52
.. _SmartSim-PR53: https://github.com/CrayLabs/SmartSim/pull/53
.. _SmartSim-PR57: https://github.com/CrayLabs/SmartSim/pull/57
.. _SmartSim-PR58: https://github.com/CrayLabs/SmartSim/pull/58
.. _SmartSim-PR60: https://github.com/CrayLabs/SmartSim/pull/60
.. _SmartSim-PR61: https://github.com/CrayLabs/SmartSim/pull/61
.. _SmartSim-PR62: https://github.com/CrayLabs/SmartSim/pull/62
.. _SmartSim-PR67: https://github.com/CrayLabs/SmartSim/pull/67
.. _SmartSim-PR69: https://github.com/CrayLabs/SmartSim/pull/69
.. _SmartSim-PR71: https://github.com/CrayLabs/SmartSim/pull/71
.. _SmartSim-PR72: https://github.com/CrayLabs/SmartSim/pull/72
.. _SmartSim-PR73: https://github.com/CrayLabs/SmartSim/pull/73

0.3.1
-----

Released on May 5, 2021

Description:
This release was dedicated to making the install process
easier. SmartSim can be installed from PyPI now and the
``smart`` cli tool makes installing the machine learning
runtimes much easier.

 - Pip install (SmartSim-PR42_)
 - ``smart`` cli tool for ML backends (SmartSim-PR42_)
 - Build Documentation for updated install (SmartSim-PR43_)
 - Migrate from Jenkins to Github Actions CI (SmartSim-PR42_)
 - Bug fix for setup.cfg (SmartSim-PR35_)

.. _SmartSim-PR43: https://github.com/CrayLabs/SmartSim/pull/43
.. _SmartSim-PR42: https://github.com/CrayLabs/SmartSim/pull/42
.. _SmartSim-PR35: https://github.com/CrayLabs/SmartSim/pull/35

0.3.0
-----

Released on April 1, 2021

Description:

 - initial 0.3.0 (first public) release of SmartSim


---------------------------------------------------------------

SmartRedis
==========

.. _changelog:

.. include:: ../smartredis/doc/changelog.rst
    :start-line: 3
