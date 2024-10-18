# Changelog

Listed here are the changes between each release of SmartSim,
SmartRedis and SmartDashboard.

Jump to:
- {ref}`SmartRedis changelog<smartredis-changelog>`
- {ref}`SmartDashboard changelog<smartdashboard-changelog>`

## SmartSim

### MLI branch

Description

- Implement asynchronous notifications for shared data
- Quick bug fix in _validate
- Add helper methods to MLI classes
- Update error handling for consistency
- Parameterize installation of dragon package with `smart build`
- Update docstrings
- Filenames conform to snake case
- Update SmartSim environment variables using new naming convention
- Refactor `exception_handler`
- Add RequestDispatcher and the possibility of batching inference requests
- Enable hostname selection for dragon tasks
- Remove pydantic dependency from MLI code
- Update MLI environment variables using new naming convention
- Reduce a copy by using torch.from_numpy instead of torch.tensor
- Enable dynamic feature store selection
- Fix dragon package installation bug
- Adjust schemas for better performance
- Add TorchWorker first implementation and mock inference app example
- Add error handling in Worker Manager pipeline
- Add EnvironmentConfigLoader for ML Worker Manager
- Add Model schema with model metadata included
- Removed device from schemas, MessageHandler and tests
- Add ML worker manager, sample worker, and feature store
- Add schemas and MessageHandler class for de/serialization of
  inference requests and response messages


### Develop

To be released at some point in the future

Description

- Implement workaround for Tensorflow that allows RedisAI to build with GCC-14
- Add instructions for installing SmartSim on PML's Scylla

Detailed Notes

- In libtensorflow, the input argument to TF_SessionRun seems to be mistyped to
  TF_Output instead of TF_Input. These two types differ only in name. GCC-14
  catches this and throws an error, even though earlier versions allow this. To
  solve this problem, patches are applied to the Tensorflow backend in RedisAI.
  Future versions of Tensorflow may fix this problem, but for now this seems to be
  the best workaround.
  ([SmartSim-PR738](https://github.com/CrayLabs/SmartSim/pull/738))
- PML's Scylla is still under development. The usual SmartSim
  build instructions do not apply because the GPU dependencies
  have yet to be installed at a system-wide level. Scylla has
  its own entry in the documentation.
  ([SmartSim-PR733](https://github.com/CrayLabs/SmartSim/pull/733))


### 0.8.0

Released on 27 September, 2024

Description

- Add instructions for Frontier to set the MIOPEN cache
- Refine Frontier documentation for proper use of miniforge3
- Refactor to the RedisAI build to allow more flexibility in versions
  and sources of ML backends
- Add Dockerfiles with GPU support
- Fine grain build support for GPUs
- Update Torch to 2.1.0, Tensorflow to 2.15.0
- Better error messages in build process
- Allow specifying Model and Ensemble parameters with
  number-like types (e.g. numpy types)
- Pin watchdog to 4.x
- Update codecov to 4.5.0
- Remove build of Redis from setup.py
- Mitigate dependency installation issues
- Fix internal host name representation for Dragon backend
- Make dependencies more discoverable in setup.py
- Add hardware pinning capability when using dragon
- Pin NumPy version to 1.x
- New launcher support for SGE (and similar derivatives)
- Fix test outputs being created in incorrect directory
- Improve support for building SmartSim without ML backends
- Update packaging dependency
- Remove broken oss.redis.com URI blocking documentation generation

Detailed Notes

- On Frontier, the MIOPEN cache may need to be set prior to using
  RedisAI in the ``smart validate``. The instructions for Frontier
  have been updated accordingly.
  ([SmartSim-PR727](https://github.com/CrayLabs/SmartSim/pull/727))
- On Frontier, the recommended way to activate conda environments is
  to go through source activate. This also means that ``conda init``
  is not needed. The instructions for Frontier have been updated to
  reflect this.
  ([SmartSim-PR719](https://github.com/CrayLabs/SmartSim/pull/719))
- The RedisAIBuilder class was completely overhauled to allow users to
  express a wider range of support for hardware/software stacks. This
  will be extended to support ROCm, CUDA-11, and CUDA-12.
  ([SmartSim-PR669](https://github.com/CrayLabs/SmartSim/pull/669))
- Versions for each of these packages are no longer specified in an
  internal class. Instead a default set of JSON files specifies the
  sources and versions. Users can specify their own custom specifications
  at smart build time.
  ([SmartSim-PR669](https://github.com/CrayLabs/SmartSim/pull/669))
- Because all build configuration has been moved to static files and all
  backends are compiled during `smart build`, SmartSim can now be shipped as a
  pure python wheel.
  ([SmartSim-PR728](https://github.com/CrayLabs/SmartSim/pull/728))
- Two new Dockerfiles are now provided (one each for 11.8 and 12.1) that
  can be used to build a container to run the tutorials. No HPC support
  should be expected at this time
  ([SmartSim-PR669](https://github.com/CrayLabs/SmartSim/pull/669))
- As a result of the previous change, SmartSim now requires C++17 and a
  minimum Cuda version of 11.8 in order to build Torch 2.1.0.
  ([SmartSim-PR669](https://github.com/CrayLabs/SmartSim/pull/669))
- Error messages were not being interpolated correctly. This has been
  addressed to provide more context when exposing error messages to users.
  ([SmartSim-PR669](https://github.com/CrayLabs/SmartSim/pull/669))
- The serializer would fail if a parameter for a Model or Ensemble
  was specified as a numpy dtype. The constructors for these
  methods now validate that the input is number-like and convert
  them to strings
  ([SmartSim-PR676](https://github.com/CrayLabs/SmartSim/pull/676))
- Pin watchdog to 4.x because v5 introduces new types and requires
  updates to the type-checking
  ([SmartSim-PR690](https://github.com/CrayLabs/SmartSim/pull/690))
- Update codecov to 4.5.0 to mitigate GitHub action failure
  ([SmartSim-PR657](https://github.com/CrayLabs/SmartSim/pull/657))
- The builder module was included in setup.py to allow us to ship the
  main Redis binaries (not RedisAI) with installs from PyPI. To
  allow easier maintenance of this file and enable future complexity
  this has been removed. The Redis binaries will thus be built
  by users during the `smart build` step
- Installation of mypy or dragon in separate build actions caused
  some dependencies (typing_extensions, numpy) to be upgraded and
  caused runtime failures. The build actions were tweaked to include
  all optional dependencies to be considered by pip during resolution.
  Additionally, the numpy version was capped on dragon installations.
  ([SmartSim-PR653](https://github.com/CrayLabs/SmartSim/pull/653))
- setup.py used to define dependencies in a way that was not amenable
  to code scanning tools. Direct dependencies now appear directly
  in the setup call and the definition of the SmartRedis version
  has been removed
  ([SmartSim-PR635](https://github.com/CrayLabs/SmartSim/pull/635))
- The separate definition of dependencies for the docs in
  requirements-doc.txt is now defined as an extra.
  ([SmartSim-PR635](https://github.com/CrayLabs/SmartSim/pull/635))
- The new major version release of Numpy is incompatible with modules
  compiled against Numpy 1.x. For both SmartSim and SmartRedis we
  request a 1.x version of numpy. This is needed in SmartSim because
  some of the downstream dependencies request NumPy
  ([SmartSim-PR623](https://github.com/CrayLabs/SmartSim/pull/623))
- SGE is now a supported launcher for SmartSim. Users can now define
  BatchSettings which will be monitored by the TaskManager. Additionally,
  if the MPI implementation was built with SGE support, Orchestrators can
  use `mpirun` without needing to specify the hosts
  ([SmartSim-PR610](https://github.com/CrayLabs/SmartSim/pull/610))
- Ensure outputs from tests are written to temporary `tests/test_output` directory
- Fix an error that would prevent ``smart build`` from moving a successfully
  compiled RedisAI shared object to the install location expected by SmartSim
  if no ML backend installations were found. Previously, this would effectively
  require users to build and install an ML backend to use the SmartSim
  orchestrator even if it was not necessary for their workflow. Users can
  install SmartSim without ML backends by running
  ``smart build --no_tf --no_pt`` and the RedisAI shared object will now be
  placed in the expected location.
  ([SmartSim-PR601](https://github.com/CrayLabs/SmartSim/pull/601))
- Fix packaging failures due to deprecated `pkg_resources`. ([SmartSim-PR598](https://github.com/CrayLabs/SmartSim/pull/598))

### 0.7.0

Released on 14 May, 2024

Description

-   Update tutorials and tutorial containers
-   Improve Dragon server shutdown
-   Add dragon runtime installer
-   Add launcher based on Dragon
-   Reuse Orchestrators within the testing suite to improve performance.
-   Fix building of documentation
-   Preview entities on experiment before start
-   Update authentication in release workflow
-   Auto-generate type-hints into documentation
-   Auto-post release PR to develop
-   Bump manifest.json to version 0.0.4
-   Fix symlinking batch ensemble and model bug
-   Fix noisy failing WLM test
-   Remove defensive regexp in .gitignore
-   Upgrade ubuntu to 22.04
-   Remove helper function `init_default`
-   Fix telemetry monitor logging errors for task history
-   Change default path for entities
-   Drop Python 3.8 support
-   Update watchdog dependency
-   Historical output files stored under .smartsim directory
-   Fixes unfalsifiable test that tests SmartSim's custom SIGINT signal
    handler
-   Add option to build Torch backend without the Intel Math Kernel
    Library
-   Fix ReadTheDocs build issue
-   Disallow uninitialized variable use
-   Promote device options to an Enum
-   Update telemetry monitor, add telemetry collectors
-   Add method to specify node features for a Slurm job
-   Colo Orchestrator setup now blocks application start until setup
    finished
-   Refactor areas of the code where mypy potential errors
-   Minor enhancements to test suite
-   ExecArgs handling correction
-   ReadTheDocs config file added and enabled on PRs
-   Enforce changelog updates
-   Fix Jupyter notebook math expressions
-   Remove deprecated SmartSim modules
-   SmartSim Documentation refactor
-   Promote SmartSim statuses to a dedicated type
-   Update the version of Redis from [7.0.4]{.title-ref} to
    [7.2.4]{.title-ref}
-   Increase disk space in doc builder container
-   Update Experiment API typing
-   Prevent duplicate entity names
-   Fix publishing of development docs

Detailed Notes

-   The tutorials are up-to date with SmartSim and SmartRedis APIs. Additionally,
    the tutorial containers' Docker files are updated. ([SmartSim-PR589](https://github.com/CrayLabs/SmartSim/pull/589))
-   The Dragon server will now terminate any process which is still running
    when a request of an immediate shutdown is sent. ([SmartSim-PR582](https://github.com/CrayLabs/SmartSim/pull/582))
-   Add `--dragon` option to `smart build`. Install appropriate Dragon
    runtime from Dragon GitHub release assets.
    ([SmartSim-PR580](https://github.com/CrayLabs/SmartSim/pull/580))
-   Add new launcher, based on [Dragon](https://dragonhpc.github.io/dragon/doc/_build/html/index.html).
    The new launcher is compatible with the Slurm and PBS schedulers and can
    be selected by specifying ``launcher="dragon"`` when creating an `Experiment`,
    or by using ``DragonRunSettings`` to launch a job. The Dragon launcher
    is at an early stage of development: early adopters are referred to the
    dedicated documentation section to learn more about it. ([SmartSim-PR580](https://github.com/CrayLabs/SmartSim/pull/580))
-   Tests may now request a given configuration and will reconnect to
    the existing orchestrator instead of building up and tearing down
    a new one each test.
    ([SmartSim-PR567](https://github.com/CrayLabs/SmartSim/pull/567))
-   Manually ensure that typing_extensions==4.6.1 in Dockerfile used to build
    docs. This fixes the deploy_dev_docs Github action ([SmartSim-PR564](https://github.com/CrayLabs/SmartSim/pull/564))
-   Added preview functionality to Experiment, including preview of all entities, active infrastructure and
    client configuration. ([SmartSim-PR525](https://github.com/CrayLabs/SmartSim/pull/525))
-   Replace the developer created token with the GH_TOKEN environment variable.
    ([SmartSim-PR570](https://github.com/CrayLabs/SmartSim/pull/570))
-   Add extension to auto-generate function type-hints into documentation.
    ([SmartSim-PR561](https://github.com/CrayLabs/SmartSim/pull/561))
-   Add to github release workflow to auto generate a pull request from
    master into develop for release.
    ([SmartSim-PR566](https://github.com/CrayLabs/SmartSim/pull/566))
-   The manifest.json version needs to match the SmartDashboard version,
    which is 0.0.4 in the upcoming release.
    ([SmartSim-PR563](https://github.com/CrayLabs/SmartSim/pull/563))
-   Properly symlinks batch ensembles and batch models.
    ([SmartSim-PR547](https://github.com/CrayLabs/SmartSim/pull/547))
-   Remove defensive regexp in .gitignore and ensure tests write to
    test_output.
    ([SmartSim-PR560](https://github.com/CrayLabs/SmartSim/pull/560))
-   After dropping support for Python 3.8, ubuntu needs to be upgraded.
    ([SmartSim-PR558](https://github.com/CrayLabs/SmartSim/pull/558))
-   Remove helper function `init_default` and replace with traditional
    type narrowing.
    ([SmartSim-PR545](https://github.com/CrayLabs/SmartSim/pull/545))
-   Ensure the telemetry monitor does not track a task_id for a managed
    task.
    ([SmartSim-PR557](https://github.com/CrayLabs/SmartSim/pull/557))
-   The default path for an entity is now the path to the experiment /
    the entity name. create_database and create_ensemble now have path
    arguments. All path arguments are compatible with relative paths.
    Relative paths are relative to the CWD.
    ([SmartSim-PR533](https://github.com/CrayLabs/SmartSim/pull/533))
-   Python 3.8 is reaching its end-of-life in October, 2024, so it will
    no longer continue to be supported.
    ([SmartSim-PR544](https://github.com/CrayLabs/SmartSim/pull/544))
-   Update watchdog dependency from 3.x to 4.x, fix new type issues
    ([SmartSim-PR540](https://github.com/CrayLabs/SmartSim/pull/540))
-   The dashboard needs to display historical logs, so log files are
    written out under the .smartsim directory and files under the
    experiment directory are symlinked to them.
    ([SmartSim-PR532](https://github.com/CrayLabs/SmartSim/pull/532))
-   Add an option to smart build
    \"\--torch_with_mkl\"/\"\--no_torch_with_mkl\" to prevent Torch from
    trying to link in the Intel Math Kernel Library. This is needed
    because on machines that have the Intel compilers installed, the
    Torch will unconditionally try to link in this library, however
    fails because the linking flags are incorrect.
    ([SmartSim-PR538](https://github.com/CrayLabs/SmartSim/pull/538))
-   Change typing\_extensions and pydantic versions in readthedocs
    environment to enable docs build.
    ([SmartSim-PR537](https://github.com/CrayLabs/SmartSim/pull/537))
-   Promote devices to a dedicated Enum type throughout the SmartSim
    code base.
    ([SmartSim-PR527](https://github.com/CrayLabs/SmartSim/pull/527))
-   Update the telemetry monitor to enable retrieval of metrics on a
    scheduled interval. Switch basic experiment tracking telemetry to
    default to on. Add database metric collectors. Improve telemetry
    monitor logging. Create telemetry subpackage at
    [smartsim.\_core.utils.telemetry]{.title-ref}. Refactor telemetry
    monitor entrypoint.
    ([SmartSim-PR460](https://github.com/CrayLabs/SmartSim/pull/460))
-   Users can now specify node features for a Slurm job through
    `SrunSettings.set_node_feature`. The method accepts a string or list
    of strings.
    ([SmartSim-PR529](https://github.com/CrayLabs/SmartSim/pull/529))
-   The request to the colocated entrypoints file within the shell
    script is now a blocking process. Once the Orchestrator is setup, it
    returns which moves the process to the background and allows the
    application to start. This prevents the application from requesting
    a ML model or script that has not been uploaded to the Orchestrator
    yet.
    ([SmartSim-PR522](https://github.com/CrayLabs/SmartSim/pull/522))
-   Add checks and tests to ensure SmartSim users cannot initialize run
    settings with a list of lists as the exe_args argument.
    ([SmartSim-PR517](https://github.com/CrayLabs/SmartSim/pull/517))
-   Add readthedocs configuration file and enable readthedocs builds on
    pull requests. Additionally added robots.txt file generation when
    readthedocs environment detected.
    ([SmartSim-PR512](https://github.com/CrayLabs/SmartSim/pull/512))
-   Add Github Actions workflow that checks if changelog is edited on
    pull requests into develop.
    ([SmartSim-PR518](https://github.com/CrayLabs/SmartSim/pull/518))
-   Add path to MathJax.js file so that Sphinx will use to render math
    expressions.
    ([SmartSim-PR516](https://github.com/CrayLabs/SmartSim/pull/516))
-   Removed deprecated SmartSim modules: slurm and mpirunSettings.
    ([SmartSim-PR514](https://github.com/CrayLabs/SmartSim/pull/514))
-   Implemented new structure of SmartSim documentation. Added examples
    images and further detail of SmartSim components.
    ([SmartSim-PR463](https://github.com/CrayLabs/SmartSim/pull/463))
-   Promote SmartSim statuses to a dedicated type named SmartSimStatus.
    ([SmartSim-PR509](https://github.com/CrayLabs/SmartSim/pull/509))
-   Update Redis version to [7.2.4]{.title-ref}. This change fixes an
    issue in the Redis build scripts causing failures on Apple Silicon
    hosts.
    ([SmartSim-PR507](https://github.com/CrayLabs/SmartSim/pull/507))
-   The container which builds the documentation for every merge to
    develop was failing due to a lack of space within the container.
    This was fixed by including an additional Github action that removes
    some unneeded software and files that come from the default Github
    Ubuntu container.
    ([SmartSim-PR504](https://github.com/CrayLabs/SmartSim/pull/504))
-   Update the generic [t.Any]{.title-ref} typehints in Experiment API.
    ([SmartSim-PR501](https://github.com/CrayLabs/SmartSim/pull/501))
-   The CI will fail static analysis if common erroneous truthy checks
    are detected.
    ([SmartSim-PR524](https://github.com/CrayLabs/SmartSim/pull/524))
-   Prevent the launch of duplicate named entities. Allow completed
    entities to run.
    ([SmartSim-PR480](https://github.com/CrayLabs/SmartSim/pull/480))
-   The CI will fail static analysis if a local variable used while
    potentially undefined.
    ([SmartSim-PR521](https://github.com/CrayLabs/SmartSim/pull/521))
-   Remove previously deprecated behavior present in test suite on
    machines with Slurm and Open MPI.
    ([SmartSim-PR520](https://github.com/CrayLabs/SmartSim/pull/520))
-   Experiments in the WLM tests are given explicit paths to prevent
    unexpected directory creation. Ensure database are not left open on
    test suite failures. Update path to pickle file in
    `tests/full_wlm/test_generic_orc_launch_batch.py::test_launch_cluster_orc_reconnect`
    to conform with changes made in
    ([SmartSim-PR533](https://github.com/CrayLabs/SmartSim/pull/533)).
    ([SmartSim-PR559](https://github.com/CrayLabs/SmartSim/pull/559))
-   When calling `Experiment.start` SmartSim would register a signal
    handler that would capture an interrupt signal (\^C) to kill any
    jobs launched through its `JobManager`. This would replace the
    default (or user defined) signal handler. SmartSim will now attempt
    to kill any launched jobs before calling the previously registered
    signal handler.
    ([SmartSim-PR535](https://github.com/CrayLabs/SmartSim/pull/535))

### 0.6.2

Released on 16 February, 2024

Description

-   Patch SmartSim dependency version

Detailed Notes

-   A critical performance concern was identified and addressed in
    SmartRedis. A patch fix was deployed, and SmartSim was updated to
    ensure users do not inadvertently pull the unpatched version of
    SmartRedis.
    ([SmartSim-PR493](https://github.com/CrayLabs/SmartSim/pull/493))

### 0.6.1

Released on 15 February, 2024

Description

-   Duplicate for DBModel/Script prevented
-   Update license to include 2024
-   Telemetry monitor is now active by default
-   Add support for Mac OSX on Apple Silicon
-   Remove Torch warnings during testing
-   Validate Slurm timing format
-   Expose Python Typehints
-   Fix test_logs to prevent generation of directory
-   Fix Python Typehint for colocated database settings
-   Python 3.11 Support
-   Quality of life [smart validate]{.title-ref} improvements
-   Remove Cobalt support
-   Enrich logging through context variables
-   Upgrade Machine Learning dependencies
-   Override sphinx-tabs background color
-   Add concurrency group to test workflow
-   Fix index when installing torch through smart build

Detailed Notes

-   Modify the [git clone]{.title-ref} for both Redis and RedisAI to set
    the line endings to unix-style line endings when using MacOS on ARM.
    ([SmartSim-PR482](https://github.com/CrayLabs/SmartSim/pull/482))
-   Separate install instructions are now provided for Mac OSX on x64 vs
    ARM64
    ([SmartSim-PR479](https://github.com/CrayLabs/SmartSim/pull/479))
-   Prevent duplicate ML model and script names being added to an
    Ensemble member if the names exists.
    ([SmartSim-PR475](https://github.com/CrayLabs/SmartSim/pull/475))
-   Updates [Copyright (c) 2021-2023]{.title-ref} to [Copyright (c)
    2021-2024]{.title-ref} in all of the necessary files.
    ([SmartSim-PR485](https://github.com/CrayLabs/SmartSim/pull/485))
-   Bug fix which prevents the expected behavior when the
    [SMARTSIM_LOG_LEVEL]{.title-ref} environment variable was set to
    [developer]{.title-ref}.
    ([SmartSim-PR473](https://github.com/CrayLabs/SmartSim/pull/473))
-   Sets the default value of the \"enable telemetry\" flag to on. Bumps
    the output [manifest.json]{.title-ref} version number to match that
    of [smartdashboard]{.title-ref} and pins a watchdog version to avoid
    build errors.
    ([SmartSim-PR477](https://github.com/CrayLabs/SmartSim/pull/477))
-   Refactor logic of [Manifest.has_db_objects]{.title-ref} to remove
    excess branching and improve readability/maintainability.
    ([SmartSim-PR476](https://github.com/CrayLabs/SmartSim/pull/476))
-   SmartSim can now be built and used on platforms using Apple Silicon
    (ARM64). Currently, only the PyTorch backend is supported. Note that
    libtorch will be downloaded from a CrayLabs github repo.
    ([SmartSim-PR465](https://github.com/CrayLabs/SmartSim/pull/465))
-   Tests that were saving Torch models were emitting warnings. These
    warnings were addressed by updating the model save test function.
    ([SmartSim-PR472](https://github.com/CrayLabs/SmartSim/pull/472))
-   Validate the timing format when requesting a slurm allocation.
    ([SmartSim-PR471](https://github.com/CrayLabs/SmartSim/pull/471))
-   Add and ship [py.typed]{.title-ref} marker to expose inline type
    hints. Fix type errors related to SmartRedis.
    ([SmartSim-PR468](https://github.com/CrayLabs/SmartSim/pull/468))
-   Fix the [test_logs.py::test_context_leak]{.title-ref} test that was
    erroneously creating a directory named [some value]{.title-ref} in
    SmartSim\'s root directory.
    ([SmartSim-PR467](https://github.com/CrayLabs/SmartSim/pull/467))
-   Add Python type hinting to colocated settings.
    ([SmartSim-PR462](https://github.com/CrayLabs/SmartSim/pull/462))
-   Add github actions for running black and isort checks.
    ([SmartSim-PR464](https://github.com/CrayLabs/SmartSim/pull/464))
-   Relax the required version of [typing_extensions]{.title-ref}.
    ([SmartSim-PR459](https://github.com/CrayLabs/SmartSim/pull/459))
-   Addition of Python 3.11 to SmartSim.
    ([SmartSim-PR461](https://github.com/CrayLabs/SmartSim/pull/461))
-   Quality of life [smart validate]{.title-ref} improvements such as
    setting [CUDA_VISIBLE_DEVICES]{.title-ref} environment variable
    within [smart validate]{.title-ref} prior to importing any ML deps
    to prevent false negatives on multi-GPU systems. Additionally, move
    SmartRedis logs from standard out to dedicated log file in the
    validation temporary directory as well as suppress
    [sklearn]{.title-ref} deprecation warning by pinning
    [KMeans]{.title-ref} constructor argument. Lastly, move TF test to
    last as TF may reserve the GPUs it uses.
    ([SmartSim-PR458](https://github.com/CrayLabs/SmartSim/pull/458))
-   Some actions in the current GitHub CI/CD workflows were outdated.
    They were replaced with the latest versions.
    ([SmartSim-PR446](https://github.com/CrayLabs/SmartSim/pull/446))
-   As the Cobalt workload manager is not used on any system we are
    aware of, its support in SmartSim was terminated and classes such as
    [CobaltLauncher]{.title-ref} have been removed.
    ([SmartSim-PR448](https://github.com/CrayLabs/SmartSim/pull/448))
-   Experiment logs are written to a file that can be read by the
    dashboard.
    ([SmartSim-PR452](https://github.com/CrayLabs/SmartSim/pull/452))
-   Updated SmartSim\'s machine learning backends to PyTorch 2.0.1,
    Tensorflow 2.13.1, ONNX 1.14.1, and ONNX Runtime 1.16.1. As a result
    of this change, there is now an available ONNX wheel for use with
    Python 3.10, and wheels for all of SmartSim\'s machine learning
    backends with Python 3.11.
    ([SmartSim-PR451](https://github.com/CrayLabs/SmartSim/pull/451))
    ([SmartSim-PR461](https://github.com/CrayLabs/SmartSim/pull/461))
-   The sphinx-tabs documentation extension uses a white background for
    the tabs component. A custom CSS for those components to inherit the
    overall theme color has been added.
    ([SmartSim-PR453](https://github.com/CrayLabs/SmartSim/pull/453))
-   Add concurrency groups to GitHub\'s CI/CD workflows, preventing
    multiple workflows from the same PR to be launched concurrently.
    ([SmartSim-PR439](https://github.com/CrayLabs/SmartSim/pull/439))
-   Torch changed their preferred indexing when trying to install their
    provided wheels. Updated the [pip install]{.title-ref} command
    within [smart build]{.title-ref} to ensure that the appropriate
    packages can be found.
    ([SmartSim-PR449](https://github.com/CrayLabs/SmartSim/pull/449))

### 0.6.0

Released on 18 December, 2023

Description

-   Conflicting directives in the SmartSim packaging instructions were
    fixed
-   [sacct]{.title-ref} and [sstat]{.title-ref} errors are now fatal for
    Slurm-based workflow executions
-   Added documentation section about ML features and TorchScript
-   Added TorchScript functions to Online Analysis tutorial
-   Added multi-DB example to documentation
-   Improved test stability on HPC systems
-   Added support for producing & consuming telemetry outputs
-   Split tests into groups for parallel execution in CI/CD pipeline
-   Change signature of [Experiment.summary()]{.title-ref}
-   Expose first_device parameter for scripts, functions, models
-   Added support for MINBATCHTIMEOUT in model execution
-   Remove support for RedisAI 1.2.5, use RedisAI 1.2.7 commit
-   Add support for multiple databases

Detailed Notes

-   Several conflicting directives between the [setup.py]{.title-ref}
    and the [setup.cfg]{.title-ref} were fixed to mitigate warnings
    issued when building the pip wheel.
    ([SmartSim-PR435](https://github.com/CrayLabs/SmartSim/pull/435))
-   When the Slurm functions [sacct]{.title-ref} and [sstat]{.title-ref}
    returned an error, it would be ignored and SmartSim\'s state could
    become inconsistent. To prevent this, errors raised by
    [sacct]{.title-ref} or [sstat]{.title-ref} now result in an
    exception.
    ([SmartSim-PR392](https://github.com/CrayLabs/SmartSim/pull/392))
-   A section named *ML Features* was added to documentation. It
    contains multiple examples of how ML models and functions can be
    added to and executed on the DB. TorchScript-based post-processing
    was added to the *Online Analysis* tutorial
    ([SmartSim-PR411](https://github.com/CrayLabs/SmartSim/pull/411))
-   An example of how to use multiple Orchestrators concurrently was
    added to the documentation
    ([SmartSim-PR409](https://github.com/CrayLabs/SmartSim/pull/409))
-   The test infrastructure was improved. Tests on HPC system are now
    stable, and issues such as non-stopped [Orchestrators]{.title-ref}
    or experiments created in the wrong paths have been fixed
    ([SmartSim-PR381](https://github.com/CrayLabs/SmartSim/pull/381))
-   A telemetry monitor was added to check updates and produce events
    for SmartDashboard
    ([SmartSim-PR426](https://github.com/CrayLabs/SmartSim/pull/426))
-   Split tests into [group_a]{.title-ref}, [group_b]{.title-ref},
    [slow_tests]{.title-ref} for parallel execution in CI/CD pipeline
    ([SmartSim-PR417](https://github.com/CrayLabs/SmartSim/pull/417),
    [SmartSim-PR424](https://github.com/CrayLabs/SmartSim/pull/424))
-   Change [format]{.title-ref} argument to [style]{.title-ref} in
    [Experiment.summary()]{.title-ref}, this is an API break
    ([SmartSim-PR391](https://github.com/CrayLabs/SmartSim/pull/391))
-   Added support for first_device parameter for scripts, functions, and
    models. This causes them to be loaded to the first num_devices
    beginning with first_device
    ([SmartSim-PR394](https://github.com/CrayLabs/SmartSim/pull/394))
-   Added support for MINBATCHTIMEOUT in model execution, which caps the
    delay waiting for a minimium number of model execution operations to
    accumulate before executing them as a batch
    ([SmartSim-PR387](https://github.com/CrayLabs/SmartSim/pull/387))
-   RedisAI 1.2.5 is not supported anymore. The only RedisAI version is
    now 1.2.7. Since the officially released RedisAI 1.2.7 has a bug
    which breaks the build process on Mac OSX, it was decided to use
    commit
    [634916c](https://github.com/RedisAI/RedisAI/commit/634916c722e718cc6ea3fad46e63f7d798f9adc2)
    from RedisAI\'s GitHub repository, where such bug has been fixed.
    This applies to all operating systems.
    ([SmartSim-PR383](https://github.com/CrayLabs/SmartSim/pull/383))
-   Add support for creation of multiple databases with unique
    identifiers.
    ([SmartSim-PR342](https://github.com/CrayLabs/SmartSim/pull/342))

### 0.5.1

Released on 14 September, 2023

Description

-   Add typehints throughout the SmartSim codebase
-   Provide support for Slurm heterogeneous jobs
-   Provide better support for [PalsMpiexecSettings]{.title-ref}
-   Allow for easier inspection of SmartSim entities
-   Log ignored error messages from [sacct]{.title-ref}
-   Fix colocated db preparation bug when using
    [JsrunSettings]{.title-ref}
-   Fix bug when user specify CPU and devices greater than 1
-   Fix bug when get_allocation called with reserved keywords
-   Enabled mypy in CI for better type safety
-   Mitigate additional suppressed pylint errors
-   Update linting support and apply to existing errors
-   Various improvements to the [smart]{.title-ref} CLI
-   Various documentation improvements
-   Various test suite improvements

Detailed Notes

-   Add methods to allow users to inspect files attached to models and
    ensembles.
    ([SmartSim-PR352](https://github.com/CrayLabs/SmartSim/pull/352))
-   Add a [smart info]{.title-ref} target to provide rudimentary
    information about the SmartSim installation.
    ([SmartSim-PR350](https://github.com/CrayLabs/SmartSim/pull/350))
-   Remove unnecessary generation producing unexpected directories in
    the test suite.
    ([SmartSim-PR349](https://github.com/CrayLabs/SmartSim/pull/349))
-   Add support for heterogeneous jobs to [SrunSettings]{.title-ref} by
    allowing users to set the [\--het-group]{.title-ref} parameter.
    ([SmartSim-PR346](https://github.com/CrayLabs/SmartSim/pull/346))
-   Provide clearer guidelines on how to contribute to SmartSim.
    ([SmartSim-PR344](https://github.com/CrayLabs/SmartSim/pull/344))
-   Integrate [PalsMpiexecSettings]{.title-ref} into the
    [Experiment]{.title-ref} factory methods when using the
    [\"pals\"]{.title-ref} launcher.
    ([SmartSim-PR343](https://github.com/CrayLabs/SmartSim/pull/343))
-   Create public properties where appropriate to mitigate
    [protected-access]{.title-ref} errors.
    ([SmartSim-PR341](https://github.com/CrayLabs/SmartSim/pull/341))
-   Fix a failure to execute [\_prep_colocated_db]{.title-ref} due to
    incorrect named attr check.
    ([SmartSim-PR339](https://github.com/CrayLabs/SmartSim/pull/339))
-   Enabled and mitigated mypy [disallow_any_generics]{.title-ref} and
    [warn_return_any]{.title-ref}.
    ([SmartSim-PR338](https://github.com/CrayLabs/SmartSim/pull/338))
-   Add a [smart validate]{.title-ref} target to provide a simple smoke
    test to assess a SmartSim build.
    ([SmartSim-PR336](https://github.com/CrayLabs/SmartSim/pull/336),
    [SmartSim-PR351](https://github.com/CrayLabs/SmartSim/pull/351))
-   Add typehints to [smartsim.\_core.launcher.step.\*]{.title-ref}.
    ([SmartSim-PR334](https://github.com/CrayLabs/SmartSim/pull/334))
-   Log errors reported from slurm WLM when attempts to retrieve status
    fail.
    ([SmartSim-PR331](https://github.com/CrayLabs/SmartSim/pull/331),
    [SmartSim-PR332](https://github.com/CrayLabs/SmartSim/pull/332))
-   Fix incorrectly formatted positional arguments in log format
    strings.
    ([SmartSim-PR330](https://github.com/CrayLabs/SmartSim/pull/330))
-   Ensure that launchers pass environment variables to unmanaged job
    steps.
    ([SmartSim-PR329](https://github.com/CrayLabs/SmartSim/pull/329))
-   Add additional tests surrounding the [RAI_PATH]{.title-ref}
    configuration environment variable.
    ([SmartSim-PR328](https://github.com/CrayLabs/SmartSim/pull/328))
-   Remove unnecessary execution of unescaped shell commands.
    ([SmartSim-PR327](https://github.com/CrayLabs/SmartSim/pull/327))
-   Add error if user calls get_allocation with reserved keywords in
    slurm get_allocation.
    ([SmartSim-PR325](https://github.com/CrayLabs/SmartSim/pull/325))
-   Add error when user requests CPU with devices greater than 1 within
    add_ml_model and add_script.
    ([SmartSim-PR324](https://github.com/CrayLabs/SmartSim/pull/324))
-   Update documentation surrounding ensemble key prefixing.
    ([SmartSim-PR322](https://github.com/CrayLabs/SmartSim/pull/322))
-   Fix formatting of the Frontier site installation.
    ([SmartSim-PR321](https://github.com/CrayLabs/SmartSim/pull/321))
-   Update pylint dependency, update .pylintrc, mitigate non-breaking
    issues, suppress api breaks.
    ([SmartSim-PR311](https://github.com/CrayLabs/SmartSim/pull/311))
-   Refactor the [smart]{.title-ref} CLI to use subparsers for better
    documentation and extension.
    ([SmartSim-PR308](https://github.com/CrayLabs/SmartSim/pull/308))

### 0.5.0

Released on 6 July, 2023

Description

A full list of changes and detailed notes can be found below:

-   Update SmartRedis dependency to v0.4.1
-   Fix tests for db models and scripts
-   Fix add_ml_model() and add_script() documentation, tests, and code
-   Remove [requirements.txt]{.title-ref} and other places where
    dependencies were defined
-   Replace [limit_app_cpus]{.title-ref} with
    [limit_db_cpus]{.title-ref} for co-located orchestrators
-   Remove wait time associated with Experiment launch summary
-   Update and rename Redis conf file
-   Migrate from redis-py-cluster to redis-py
-   Update full test suite to not require a TF wheel at test time
-   Update doc strings
-   Remove deprecated code
-   Relax the coloredlogs version
-   Update Fortran tutorials for SmartRedis
-   Add support for multiple network interface binding in Orchestrator
    and Colocated DBs
-   Add typehints and static analysis

Detailed notes

-   Updates SmartRedis to the most current release
    ([SmartSim-PR316](https://github.com/CrayLabs/SmartSim/pull/316))
-   Fixes and enhancements to documentation
    ([SmartSim-PR317](https://github.com/CrayLabs/SmartSim/pull/317),
    [SmartSim-PR314](https://github.com/CrayLabs/SmartSim/pull/314),
    [SmartSim-PR287](https://github.com/CrayLabs/SmartSim/pull/287))
-   Various fixes and enhancements to the test suite
    ([SmartSim-PR315](https://github.com/CrayLabs/SmartSim/pull/314),
    [SmartSim-PR312](https://github.com/CrayLabs/SmartSim/pull/312),
    [SmartSim-PR310](https://github.com/CrayLabs/SmartSim/pull/310),
    [SmartSim-PR302](https://github.com/CrayLabs/SmartSim/pull/302),
    [SmartSim-PR283](https://github.com/CrayLabs/SmartSim/pull/283))
-   Fix a defect in the tests related to database models and scripts
    that was causing key collisions when testing on workload managers
    ([SmartSim-PR313](https://github.com/CrayLabs/SmartSim/pull/313))
-   Remove [requirements.txt]{.title-ref} and other places where
    dependencies were defined.
    ([SmartSim-PR307](https://github.com/CrayLabs/SmartSim/pull/307))
-   Fix defect where dictionaries used to create run settings can be
    changed unexpectedly due to copy-by-ref
    ([SmartSim-PR305](https://github.com/CrayLabs/SmartSim/pull/305))
-   The underlying code for Model.add_ml_model() and Model.add_script()
    was fixed to correctly handle multi-GPU configurations. Tests were
    updated to run on non-local launchers. Documentation was updated and
    fixed. Also, the default testing interface has been changed to lo
    instead of ipogif.
    ([SmartSim-PR304](https://github.com/CrayLabs/SmartSim/pull/304))
-   Typehints have been added. A makefile target [make
    check-mypy]{.title-ref} executes static analysis with mypy.
    ([SmartSim-PR295](https://github.com/CrayLabs/SmartSim/pull/295),
    [SmartSim-PR301](https://github.com/CrayLabs/SmartSim/pull/301),
    [SmartSim-PR303](https://github.com/CrayLabs/SmartSim/pull/303))
-   Replace [limit_app_cpus]{.title-ref} with
    [limit_db_cpus]{.title-ref} for co-located orchestrators. This
    resolves some incorrect behavior/assumptions about how the
    application would be pinned. Instead, users should directly specify
    the binding options in their application using the options
    appropriate for their launcher
    ([SmartSim-PR306](https://github.com/CrayLabs/SmartSim/pull/306))
-   Simplify code in [random_permutations]{.title-ref} parameter
    generation strategy
    ([SmartSim-PR300](https://github.com/CrayLabs/SmartSim/pull/300))
-   Remove wait time associated with Experiment launch summary
    ([SmartSim-PR298](https://github.com/CrayLabs/SmartSim/pull/298))
-   Update Redis conf file to conform with Redis v7.0.5 conf file
    ([SmartSim-PR293](https://github.com/CrayLabs/SmartSim/pull/293))
-   Migrate from redis-py-cluster to redis-py for cluster status checks
    ([SmartSim-PR292](https://github.com/CrayLabs/SmartSim/pull/292))
-   Update full test suite to no longer require a tensorflow wheel to be
    available at test time.
    ([SmartSim-PR291](https://github.com/CrayLabs/SmartSim/pull/291))
-   Correct spelling of colocated in doc strings
    ([SmartSim-PR290](https://github.com/CrayLabs/SmartSim/pull/290))
-   Deprecated launcher-specific orchestrators, constants, and ML
    utilities were removed.
    ([SmartSim-PR289](https://github.com/CrayLabs/SmartSim/pull/289))
-   Relax the coloredlogs version to be greater than 10.0
    ([SmartSim-PR288](https://github.com/CrayLabs/SmartSim/pull/288))
-   Update the Github Actions runner image from
    [macos-10.15]{.title-ref}[ to \`macos-12]{.title-ref}\`. The former
    began deprecation in May 2022 and was finally removed in May 2023.
    ([SmartSim-PR285](https://github.com/CrayLabs/SmartSim/pull/285))
-   The Fortran tutorials had not been fully updated to show how to
    handle return/error codes. These have now all been updated.
    ([SmartSim-PR284](https://github.com/CrayLabs/SmartSim/pull/284))
-   Orchestrator and Colocated DB now accept a list of interfaces to
    bind to. The argument name is still [interface]{.title-ref} for
    backward compatibility reasons.
    ([SmartSim-PR281](https://github.com/CrayLabs/SmartSim/pull/281))
-   Typehints have been added to public APIs. A makefile target to
    execute static analysis with mypy is available [make
    check-mypy]{.title-ref}.
    ([SmartSim-PR295](https://github.com/CrayLabs/SmartSim/pull/295))

### 0.4.2

Released on April 12, 2023

Description

This release of SmartSim had a focus on polishing and extending exiting
features already provided by SmartSim. Most notably, this release
provides support to allow users to colocate their models with an
orchestrator using Unix domain sockets and support for launching models
as batch jobs.

Additionally, SmartSim has updated its tool chains to provide a better
user experience. Notably, SmarSim can now be used with Python 3.10,
Redis 7.0.5, and RedisAI 1.2.7. Furthermore, SmartSim now utilizes
SmartRedis\'s aggregation lists to streamline the use and extension of
ML data loaders, making working with popular machine learning frameworks
in SmartSim a breeze.

A full list of changes and detailed notes can be found below:

-   Add support for colocating an orchestrator over UDS
-   Add support for Python 3.10, deprecate support for Python 3.7 and
    RedisAI 1.2.3
-   Drop support for Ray
-   Update ML data loaders to make use of SmartRedis\'s aggregation
    lists
-   Allow for models to be launched independently as batch jobs
-   Update to current version of Redis to 7.0.5
-   Add support for RedisAI 1.2.7, pyTorch 1.11.0, Tensorflow 2.8.0,
    ONNXRuntime 1.11.1
-   Fix bug in colocated database entrypoint when loading PyTorch models
-   Fix test suite behavior with environment variables

Detailed Notes

-   Running some tests could result in some SmartSim-specific
    environment variables to be set. Such environment variables are now
    reset after each test execution. Also, a warning for environment
    variable usage in Slurm was added, to make the user aware in case an
    environment variable will not be assigned the desired value with
    [\--export]{.title-ref}.
    ([SmartSim-PR270](https://github.com/CrayLabs/SmartSim/pull/270))
-   The PyTorch and TensorFlow data loaders were update to make use of
    aggregation lists. This breaks their API, but makes them easier to
    use.
    ([SmartSim-PR264](https://github.com/CrayLabs/SmartSim/pull/264))
-   The support for Ray was dropped, as its most recent versions caused
    problems when deployed through SmartSim. We plan to release a
    separate add-on library to accomplish the same results. If you are
    interested in getting the Ray launch functionality back in your
    workflow, please get in touch with us!
    ([SmartSim-PR263](https://github.com/CrayLabs/SmartSim/pull/263))
-   Update from Redis version 6.0.8 to 7.0.5.
    ([SmartSim-PR258](https://github.com/CrayLabs/SmartSim/pull/258))
-   Adds support for Python 3.10 without the ONNX machine learning
    backend. Deprecates support for Python 3.7 as it will stop receiving
    security updates. Deprecates support for RedisAI 1.2.3. Update the
    build process to be able to correctly fetch supported dependencies.
    If a user attempts to build an unsupported dependency, an error
    message is shown highlighting the discrepancy.
    ([SmartSim-PR256](https://github.com/CrayLabs/SmartSim/pull/256))
-   Models were given a [batch_settings]{.title-ref} attribute. When
    launching a model through [Experiment.start]{.title-ref} the
    [Experiment]{.title-ref} will first check for a non-nullish value at
    that attribute. If the check is satisfied, the
    [Experiment]{.title-ref} will attempt to wrap the underlying run
    command in a batch job using the object referenced at
    [Model.batch_settings]{.title-ref} as the batch settings for the
    job. If the check is not satisfied, the [Model]{.title-ref} is
    launched in the traditional manner as a job step.
    ([SmartSim-PR245](https://github.com/CrayLabs/SmartSim/pull/245))
-   Fix bug in colocated database entrypoint stemming from uninitialized
    variables. This bug affects PyTorch models being loaded into the
    database.
    ([SmartSim-PR237](https://github.com/CrayLabs/SmartSim/pull/237))
-   The release of RedisAI 1.2.7 allows us to update support for recent
    versions of PyTorch, Tensorflow, and ONNX
    ([SmartSim-PR234](https://github.com/CrayLabs/SmartSim/pull/234))
-   Make installation of correct Torch backend more reliable according
    to instruction from PyTorch
-   In addition to TCP, add UDS support for colocating an orchestrator
    with models. Methods [Model.colocate_db_tcp]{.title-ref} and
    [Model.colocate_db_uds]{.title-ref} were added to expose this
    functionality. The [Model.colocate_db]{.title-ref} method remains
    and uses TCP for backward compatibility
    ([SmartSim-PR246](https://github.com/CrayLabs/SmartSim/pull/246))

### 0.4.1

Released on June 24, 2022

Description: This release of SmartSim introduces a new experimental
feature to help make SmartSim workflows more portable: the ability to
run simulations models in a container via Singularity. This feature has
been tested on a small number of platforms and we encourage users to
provide feedback on its use.

We have also made improvements in a variety of areas: new utilities to
load scripts and machine learning models into the database directly from
SmartSim driver scripts and install-time choice to use either
[KeyDB]{.title-ref} or [Redis]{.title-ref} for the Orchestrator. The
[RunSettings]{.title-ref} API is now more consistent across subclasses.
Another key focus of this release was to aid new SmartSim users by
including more extensive tutorials and improving the documentation. The
docker image containing the SmartSim tutorials now also includes a
tutorial on online training.

Launcher improvements

-   New methods for specifying [RunSettings]{.title-ref} parameters
    ([SmartSim-PR166](https://github.com/CrayLabs/SmartSim/pull/166))
    ([SmartSim-PR170](https://github.com/CrayLabs/SmartSim/pull/170))
-   Better support for [mpirun]{.title-ref}, [mpiexec]{.title-ref},
    and [orterun]{.title-ref} as launchers
    ([SmartSim-PR186](https://github.com/CrayLabs/SmartSim/pull/186))
-   Experimental: add support for running models via Singularity
    ([SmartSim-PR204](https://github.com/CrayLabs/SmartSim/pull/204))

Documentation and tutorials

-   Tutorial updates
    ([SmartSim-PR155](https://github.com/CrayLabs/SmartSim/pull/155))
    ([SmartSim-PR203](https://github.com/CrayLabs/SmartSim/pull/203))
    ([SmartSim-PR208](https://github.com/CrayLabs/SmartSim/pull/208))
-   Add SmartSim Zoo info to documentation
    ([SmartSim-PR175](https://github.com/CrayLabs/SmartSim/pull/175))
-   New tutorial for demonstrating online training
    ([SmartSim-PR176](https://github.com/CrayLabs/SmartSim/pull/176))
    ([SmartSim-PR188](https://github.com/CrayLabs/SmartSim/pull/188))

General improvements and bug fixes

-   Set models and scripts at the driver level
    ([SmartSim-PR185](https://github.com/CrayLabs/SmartSim/pull/185))
-   Optionally use KeyDB for the orchestrator
    ([SmartSim-PR180](https://github.com/CrayLabs/SmartSim/pull/180))
-   Ability to specify system-level libraries
    ([SmartSim-PR154](https://github.com/CrayLabs/SmartSim/pull/154))
    ([SmartSim-PR182](https://github.com/CrayLabs/SmartSim/pull/182))
-   Fix the handling of LSF gpus_per_shard
    ([SmartSim-PR164](https://github.com/CrayLabs/SmartSim/pull/164))
-   Fix error when re-running [smart build]{.title-ref}
    ([SmartSim-PR165](https://github.com/CrayLabs/SmartSim/pull/165))
-   Fix generator hanging when tagged configuration variables are
    missing
    ([SmartSim-PR177](https://github.com/CrayLabs/SmartSim/pull/177))

Dependency updates

-   CMake version from 3.10 to 3.13
    ([SmartSim-PR152](https://github.com/CrayLabs/SmartSim/pull/152))
-   Update click to 8.0.2
    ([SmartSim-PR200](https://github.com/CrayLabs/SmartSim/pull/200))

### 0.4.0

Released on Feb 11, 2022

Description: In this release SmartSim continues to promote ease of use.
To this end SmartSim has introduced new portability features that allow
users to abstract away their targeted hardware, while providing even
more compatibility with existing libraries.

A new feature, Co-located orchestrator deployments has been added which
provides scalable online inference capabilities that overcome previous
performance limitations in seperated orchestrator/application
deployments. For more information on advantages of co-located
deployments, see the Orchestrator section of the SmartSim documentation.

The SmartSim build was significantly improved to increase customization
of build toolchain and the `smart` command line inferface was expanded.

Additional tweaks and upgrades have also been made to ensure an optimal
experience. Here is a comprehensive list of changes made in SmartSim
0.4.0.

Orchestrator Enhancements:

-   Add Orchestrator Co-location
    ([SmartSim-PR139](https://github.com/CrayLabs/SmartSim/pull/139))
-   Add Orchestrator configuration file edit methods
    ([SmartSim-PR109](https://github.com/CrayLabs/SmartSim/pull/109))

Emphasize Driver Script Portability:

-   Add ability to create run settings through an experiment
    ([SmartSim-PR110](https://github.com/CrayLabs/SmartSim/pull/110))
-   Add ability to create batch settings through an experiment
    ([SmartSim-PR112](https://github.com/CrayLabs/SmartSim/pull/112))
-   Add automatic launcher detection to experiment portability
    functions
    ([SmartSim-PR120](https://github.com/CrayLabs/SmartSim/pull/120))

Expand Machine Learning Library Support:

-   Data loaders for online training in Keras/TF and Pytorch
    ([SmartSim-PR115](https://github.com/CrayLabs/SmartSim/pull/115))
    ([SmartSim-PR140](https://github.com/CrayLabs/SmartSim/pull/140))
-   ML backend versions updated with expanded support for multiple
    versions
    ([SmartSim-PR122](https://github.com/CrayLabs/SmartSim/pull/122))
-   Launch Ray internally using `RunSettings`
    ([SmartSim-PR118](https://github.com/CrayLabs/SmartSim/pull/118))
-   Add Ray cluster setup and deployment to SmartSim
    ([SmartSim-PR50](https://github.com/CrayLabs/SmartSim/pull/50))

Expand Launcher Setting Options:

-   Add ability to use base `RunSettings` on a Slurm, or PBS launchers
    ([SmartSim-PR90](https://github.com/CrayLabs/SmartSim/pull/90))
-   Add ability to use base `RunSettings` on LFS launcher
    ([SmartSim-PR108](https://github.com/CrayLabs/SmartSim/pull/108))

Deprecations and Breaking Changes

-   Orchestrator classes combined into single implementation for
    portability
    ([SmartSim-PR139](https://github.com/CrayLabs/SmartSim/pull/139))
-   `smartsim.constants` changed to `smartsim.status`
    ([SmartSim-PR122](https://github.com/CrayLabs/SmartSim/pull/122))
-   `smartsim.tf` migrated to `smartsim.ml.tf`
    ([SmartSim-PR115](https://github.com/CrayLabs/SmartSim/pull/115))
    ([SmartSim-PR140](https://github.com/CrayLabs/SmartSim/pull/140))
-   TOML configuration option removed in favor of environment variable
    approach
    ([SmartSim-PR122](https://github.com/CrayLabs/SmartSim/pull/122))

General Improvements and Bug Fixes:

-   Improve and extend parameter handling
    ([SmartSim-PR107](https://github.com/CrayLabs/SmartSim/pull/107))
    ([SmartSim-PR119](https://github.com/CrayLabs/SmartSim/pull/119))
-   Abstract away non-user facing implementation details
    ([SmartSim-PR122](https://github.com/CrayLabs/SmartSim/pull/122))
-   Add various dimensions to the CI build matrix for SmartSim testing
    ([SmartSim-PR130](https://github.com/CrayLabs/SmartSim/pull/130))
-   Add missing functions to LSFSettings API
    ([SmartSim-PR113](https://github.com/CrayLabs/SmartSim/pull/113))
-   Add RedisAI checker for installed backends
    ([SmartSim-PR137](https://github.com/CrayLabs/SmartSim/pull/137))
-   Remove heavy and unnecessary dependencies
    ([SmartSim-PR116](https://github.com/CrayLabs/SmartSim/pull/116))
    ([SmartSim-PR132](https://github.com/CrayLabs/SmartSim/pull/132))
-   Fix LSFLauncher and LSFOrchestrator
    ([SmartSim-PR86](https://github.com/CrayLabs/SmartSim/pull/86))
-   Fix over greedy Workload Manager Parsers
    ([SmartSim-PR95](https://github.com/CrayLabs/SmartSim/pull/95))
-   Fix Slurm handling of comma-separated env vars
    ([SmartSim-PR104](https://github.com/CrayLabs/SmartSim/pull/104))
-   Fix internal method calls
    ([SmartSim-PR138](https://github.com/CrayLabs/SmartSim/pull/138))

Documentation Updates:

-   Updates to documentation build process
    ([SmartSim-PR133](https://github.com/CrayLabs/SmartSim/pull/133))
    ([SmartSim-PR143](https://github.com/CrayLabs/SmartSim/pull/143))
-   Updates to documentation content
    ([SmartSim-PR96](https://github.com/CrayLabs/SmartSim/pull/96))
    ([SmartSim-PR129](https://github.com/CrayLabs/SmartSim/pull/129))
    ([SmartSim-PR136](https://github.com/CrayLabs/SmartSim/pull/136))
    ([SmartSim-PR141](https://github.com/CrayLabs/SmartSim/pull/141))
-   Update SmartSim Examples
    ([SmartSim-PR68](https://github.com/CrayLabs/SmartSim/pull/68))
    ([SmartSim-PR100](https://github.com/CrayLabs/SmartSim/pull/100))

### 0.3.2

Released on August 10, 2021

Description:

-   Upgraded RedisAI backend to 1.2.3
    ([SmartSim-PR69](https://github.com/CrayLabs/SmartSim/pull/69))
-   PyTorch 1.7.1, TF 2.4.2, and ONNX 1.6-7
    ([SmartSim-PR69](https://github.com/CrayLabs/SmartSim/pull/69))
-   LSF launcher for IBM machines
    ([SmartSim-PR62](https://github.com/CrayLabs/SmartSim/pull/62))
-   Improved code coverage by adding more unit tests
    ([SmartSim-PR53](https://github.com/CrayLabs/SmartSim/pull/53))
-   Orchestrator methods to get address and check status
    ([SmartSim-PR60](https://github.com/CrayLabs/SmartSim/pull/60))
-   Added Manifest object that tracks deployables in Experiments
    ([SmartSim-PR61](https://github.com/CrayLabs/SmartSim/pull/61))
-   Bug fixes
    ([SmartSim-PR52](https://github.com/CrayLabs/SmartSim/pull/52))
    ([SmartSim-PR58](https://github.com/CrayLabs/SmartSim/pull/58))
    ([SmartSim-PR67](https://github.com/CrayLabs/SmartSim/pull/67))
    ([SmartSim-PR73](https://github.com/CrayLabs/SmartSim/pull/73))
-   Updated documentation and examples
    ([SmartSim-PR51](https://github.com/CrayLabs/SmartSim/pull/51))
    ([SmartSim-PR57](https://github.com/CrayLabs/SmartSim/pull/57))
    ([SmartSim-PR71](https://github.com/CrayLabs/SmartSim/pull/71))
-   Improved IP address aquisition
    ([SmartSim-PR72](https://github.com/CrayLabs/SmartSim/pull/72))
-   Binding database to network interfaces

### 0.3.1

Released on May 5, 2021

Description: This release was dedicated to making the install process
easier. SmartSim can be installed from PyPI now and the `smart` cli tool
makes installing the machine learning runtimes much easier.

-   Pip install
    ([SmartSim-PR42](https://github.com/CrayLabs/SmartSim/pull/42))
-   `smart` cli tool for ML backends
    ([SmartSim-PR42](https://github.com/CrayLabs/SmartSim/pull/42))
-   Build Documentation for updated install
    ([SmartSim-PR43](https://github.com/CrayLabs/SmartSim/pull/43))
-   Migrate from Jenkins to Github Actions CI
    ([SmartSim-PR42](https://github.com/CrayLabs/SmartSim/pull/42))
-   Bug fix for setup.cfg
    ([SmartSim-PR35](https://github.com/CrayLabs/SmartSim/pull/35))

### 0.3.0

Released on April 1, 2021

Description:

-   initial 0.3.0 (first public) release of SmartSim

------------------------------------------------------------------------

(smartredis-changelog)=
## SmartRedis

```{include} ../smartredis/doc/changelog.md
:start-line: 2
```

------------------------------------------------------------------------

(smartdashboard-changelog)=
## SmartDashboard

```{include} ../smartdashboard/doc/changelog.md
:start-line: 2
```
