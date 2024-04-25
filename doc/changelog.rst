*********
Changelog
*********

Listed here are the changes between each release of SmartSim
and SmartRedis.

Jump to :ref:`SmartRedis Changelog <sr_changelog>`


SmartSim
========

Development branch
------------------

To be released at some future point in time

Description

- Auto-generate type-hints into documentation
- Auto-post release PR to develop
- Bump manifest.json to version 0.0.4
- Fix symlinking batch ensemble and model bug
- Remove defensive regexp in .gitignore
- Upgrade ubuntu to 22.04
- Remove helper function ``init_default``
- Fix telemetry monitor logging errrors for task history
- Change default path for entities
- Drop Python 3.8 support
- Update watchdog dependency
- Historical output files stored under .smartsim directory
- Add option to build Torch backend without the Intel Math Kernel Library
- Fix ReadTheDocs build issue
- Promote device options to an Enum
- Update telemetry monitor, add telemetry collectors
- Add method to specify node features for a Slurm job
- Colo Orchestrator setup now blocks application start until setup finished
- ExecArgs handling correction
- ReadTheDocs config file added and enabled on PRs
- Enforce changelog updates
- Remove deprecated SmartSim modules
- SmartSim Documentation refactor
- Update the version of Redis from `7.0.4` to `7.2.4`
- Fix publishing of development docs
- Update Experiment API typing
- Minor enhancements to test suite
- Improve SmartSim experiment signal handlers

Detailed Notes

- Add extension to auto-generate function type-hints into documentation.
  (SmartSim-PR561_)
- Add to github release workflow to auto generate a pull request from master
  into develop for release. (SmartSim-PR566_)
- The manifest.json version needs to match the SmartDashboard version, which is
  0.0.4 in the upcoming release. (SmartSim-PR563_)
- Properly symlinks batch ensembles and batch models. (SmartSim-PR547_)
- Remove defensive regexp in .gitignore and ensure tests write to test_output.
  (SmartSim-PR560_)
- After dropping support for Python 3.8, ubuntu needs to be upgraded.
  (SmartSim-PR558_)
- Remove helper function ``init_default`` and replace with traditional type
  narrowing. (SmartSim-PR545_)
- Ensure the telemetry monitor does not track a task_id 
  for a managed task. (SmartSim-PR557_)
- The default path for an entity is now the path to the experiment / the
  entity name. create_database and create_ensemble now have path arguments.
  All path arguments are compatible with relative paths. Relative paths are
  relative to the CWD. (SmartSim-PR533_)
- Python 3.8 is reaching its end-of-life in October, 2024, so it will
  no longer continue to be supported. (SmartSim-PR544_) 
- Update watchdog dependency from 3.x to 4.x, fix new type issues (SmartSim-PR540_)
- The dashboard needs to display historical logs, so log files are written
  out under the .smartsim directory and files under the experiment
  directory are symlinked to them. (SmartSim-PR532_)
- Add an option to smart build "--torch_with_mkl"/"--no_torch_with_mkl" to
  prevent Torch from trying to link in the Intel Math Kernel Library. This
  is needed because on machines that have the Intel compilers installed, the
  Torch will unconditionally try to link in this library, however fails
  because the linking flags are incorrect. (SmartSim-PR538_)
- Change type_extension and pydantic versions in readthedocs environment
  to enable docs build. (SmartSim-PR537_)
- Promote devices to a dedicated Enum type throughout the SmartSim code base.
  (SmartSim-PR498_)
- Update the telemetry monitor to enable retrieval of metrics on a scheduled
  interval. Switch basic experiment tracking telemetry to default to on. Add
  database metric collectors. Improve telemetry monitor logging. Create
  telemetry subpackage at `smartsim._core.utils.telemetry`. Refactor
  telemetry monitor entrypoint. (SmartSim-PR460_)
- Users can now specify node features for a Slurm job through
  ``SrunSettings.set_node_feature``. The method accepts a string
  or list of strings. (SmartSim-PR529_)
- The request to the colocated entrypoints file within the shell script
  is now a blocking process. Once the Orchestrator is setup, it returns
  which moves the process to the background and allows the application to
  start. This prevents the application from requesting a ML model or
  script that has not been uploaded to the Orchestrator yet. (SmartSim-PR522_)
- Add checks and tests to ensure SmartSim users cannot initialize run settings
  with a list of lists as the exe_args argument. (SmartSim-PR517_)
- Add readthedocs configuration file and enable readthedocs builds
  on pull requests. Additionally added robots.txt file generation
  when readthedocs environment detected. (SmartSim-PR512_)
- Add Github Actions workflow that checks if changelog is edited
  on pull requests into develop. (SmartSim-PR518_)
- Removed deprecated SmartSim modules: slurm and mpirunSettings.
  (SmartSim-PR514_)
- Implemented new structure of SmartSim documentation. Added examples
  images and further detail of SmartSim components. (SmartSim-PR463_)
- Update Redis version to `7.2.4`. This change fixes an issue in the Redis
  build scripts causing failures on Apple Silicon hosts. (SmartSim-PR507_)
- The container which builds the documentation for every merge to develop
  was failing due to a lack of space within the container. This was fixed
  by including an additional Github action that removes some unneeded
  software and files that come from the default Github Ubuntu container.
  (SmartSim-PR504_)
- Update the generic `t.Any` typehints in Experiment API. (SmartSim-PR501_)
- The CI will fail static analysis if common erroneous truthy checks are
  detected. (SmartSim-PR524_)
- The CI will fail static analysis if a local variable used while potentially
  undefined. (SmartSim-PR521_)
- Remove previously deprecated behavior present in test suite on machines with
  Slurm and Open MPI. (SmartSim-PR520_)
- Experiments in the WLM tests are given explicit paths to prevent unexpected
  directory creation. Ensure database are not left open on test suite failures.
  Update path to pickle file in
  ``tests/full_wlm/test_generic_orc_launch_batch.py::test_launch_cluster_orc_reconnect``
  to conform with changes made in SmartSim-PR533_. (SmartSim-PR559_)
- When calling ``Experiment.start`` SmartSim would register a signal handler
  that would capture an interrupt signal (^C) to kill any jobs launched through
  its ``JobManager``. This would replace the default (or user defined) signal
  handler. SmartSim will now attempt to kill any launched jobs before calling
  the previously registered signal handler. (SmartSim-PR535_)

.. _SmartSim-PR561: https://github.com/CrayLabs/SmartSim/pull/561
.. _SmartSim-PR566: https://github.com/CrayLabs/SmartSim/pull/566
.. _SmartSim-PR563: https://github.com/CrayLabs/SmartSim/pull/563
.. _SmartSim-PR547: https://github.com/CrayLabs/SmartSim/pull/547
.. _SmartSim-PR560: https://github.com/CrayLabs/SmartSim/pull/560
.. _SmartSim-PR559: https://github.com/CrayLabs/SmartSim/pull/559
.. _SmartSim-PR558: https://github.com/CrayLabs/SmartSim/pull/558
.. _SmartSim-PR545: https://github.com/CrayLabs/SmartSim/pull/545
.. _SmartSim-PR557: https://github.com/CrayLabs/SmartSim/pull/557
.. _SmartSim-PR533: https://github.com/CrayLabs/SmartSim/pull/533
.. _SmartSim-PR544: https://github.com/CrayLabs/SmartSim/pull/544
.. _SmartSim-PR540: https://github.com/CrayLabs/SmartSim/pull/540
.. _SmartSim-PR532: https://github.com/CrayLabs/SmartSim/pull/532
.. _SmartSim-PR538: https://github.com/CrayLabs/SmartSim/pull/538
.. _SmartSim-PR537: https://github.com/CrayLabs/SmartSim/pull/537
.. _SmartSim-PR498: https://github.com/CrayLabs/SmartSim/pull/498
.. _SmartSim-PR460: https://github.com/CrayLabs/SmartSim/pull/460
.. _SmartSim-PR512: https://github.com/CrayLabs/SmartSim/pull/512
.. _SmartSim-PR535: https://github.com/CrayLabs/SmartSim/pull/535
.. _SmartSim-PR529: https://github.com/CrayLabs/SmartSim/pull/529
.. _SmartSim-PR522: https://github.com/CrayLabs/SmartSim/pull/522
.. _SmartSim-PR521: https://github.com/CrayLabs/SmartSim/pull/521
.. _SmartSim-PR524: https://github.com/CrayLabs/SmartSim/pull/524
.. _SmartSim-PR520: https://github.com/CrayLabs/SmartSim/pull/520
.. _SmartSim-PR518: https://github.com/CrayLabs/SmartSim/pull/518
.. _SmartSim-PR517: https://github.com/CrayLabs/SmartSim/pull/517
.. _SmartSim-PR514: https://github.com/CrayLabs/SmartSim/pull/514
.. _SmartSim-PR512: https://github.com/CrayLabs/SmartSim/pull/512
.. _SmartSim-PR507: https://github.com/CrayLabs/SmartSim/pull/507
.. _SmartSim-PR504: https://github.com/CrayLabs/SmartSim/pull/504
.. _SmartSim-PR501: https://github.com/CrayLabs/SmartSim/pull/501
.. _SmartSim-PR463: https://github.com/CrayLabs/SmartSim/pull/463


0.6.2
-----

Released on 16 February, 2024

Description

- Patch SmartSim dependency version


Detailed Notes

- A critical performance concern was identified and addressed in SmartRedis. A
  patch fix was deployed, and SmartSim was updated to ensure users do not
  inadvertently pull the unpatched version of SmartRedis. (SmartSim-PR493_)


.. _SmartSim-PR493: https://github.com/CrayLabs/SmartSim/pull/493


0.6.1
-----

Released on 15 February, 2024

Description

- Duplicate for DBModel/Script prevented
- Update license to include 2024
- Telemetry monitor is now active by default
- Add support for Mac OSX on Apple Silicon
- Remove Torch warnings during testing
- Validate Slurm timing format
- Expose Python Typehints
- Fix test_logs to prevent generation of directory
- Fix Python Typehint for colocated database settings
- Python 3.11 Support
- Quality of life `smart validate` improvements
- Remove Cobalt support
- Enrich logging through context variables
- Upgrade Machine Learning dependencies
- Override sphinx-tabs background color
- Add concurrency group to test workflow
- Fix index when installing torch through smart build


Detailed Notes

- Modify the `git clone` for both Redis and RedisAI to set the line endings to
  unix-style line endings when using MacOS on ARM. (SmartSim-PR482_)
- Separate install instructions are now provided for Mac OSX on x64 vs ARM64 (SmartSim-PR479_)
- Prevent duplicate ML model and script names being added to an
  Ensemble member if the names exists. (SmartSim-PR475_)
- Updates `Copyright (c) 2021-2023` to `Copyright (c) 2021-2024`
  in all of the necessary files. (SmartSim-PR485_)
- Bug fix which prevents the expected behavior when the `SMARTSIM_LOG_LEVEL`
  environment variable was set to `developer`. (SmartSim-PR473_)
- Sets the default value of the "enable telemetry" flag to on.
  Bumps the output `manifest.json` version number to match that of
  `smartdashboard` and pins a watchdog version to avoid build errors.
  (SmartSim-PR477_)
- Refactor logic of `Manifest.has_db_objects` to remove excess branching
  and improve readability/maintainability. (SmartSim-PR476_)
- SmartSim can now be built and used on platforms using Apple Silicon
  (ARM64). Currently, only the PyTorch backend is supported. Note that libtorch
  will be downloaded from a CrayLabs github repo. (SmartSim-PR465_)
- Tests that were saving Torch models were emitting warnings.  These warnings
  were addressed by updating the model save test function. (SmartSim-PR472_)
- Validate the timing format when requesting a slurm allocation. (SmartSim-PR471_)
- Add and ship `py.typed` marker to expose inline type hints. Fix
  type errors related to SmartRedis. (SmartSim-PR468_)
- Fix the `test_logs.py::test_context_leak` test that was
  erroneously creating a directory named `some value` in SmartSim's root
  directory. (SmartSim-PR467_)
- Add Python type hinting to colocated settings. (SmartSim-PR462_)
- Add github actions for running black and isort checks. (SmartSim-PR464_)
- Relax the required version of `typing_extensions`. (SmartSim-PR459_)
- Addition of Python 3.11 to SmartSim. (SmartSim-PR461_)
- Quality of life `smart validate` improvements such as setting `CUDA_VISIBLE_DEVICES`
  environment variable within `smart validate` prior to importing any ML deps to
  prevent false negatives on multi-GPU systems. Additionally, move SmartRedis logs
  from standard out to dedicated log file in the validation temporary directory as well as
  suppress `sklearn` deprecation warning by pinning `KMeans` constructor
  argument. Lastly, move TF test to last as TF may reserve the GPUs it uses.
  (SmartSim-PR458_)
- Some actions in the current GitHub CI/CD workflows were outdated. They were
  replaced with the latest versions. (SmartSim-PR446_)
- As the Cobalt workload manager is not used on any system we are aware of,
  its support in SmartSim was terminated and classes such as `CobaltLauncher` have
  been removed. (SmartSim-PR448_)
- Experiment logs are written to a file that can be read by the dashboard. (SmartSim-PR452_)
- Updated SmartSim's machine learning backends to PyTorch 2.0.1, Tensorflow
  2.13.1, ONNX 1.14.1, and ONNX Runtime 1.16.1. As a result of this change,
  there is now an available ONNX wheel for use with Python 3.10, and wheels for
  all of SmartSim's machine learning backends with Python 3.11.
  (SmartSim-PR451_) (SmartSim-PR461_)
- The sphinx-tabs documentation extension uses a white background for the tabs component.
  A custom CSS for those components to inherit the overall theme color has
  been added. (SmartSim-PR453_)
- Add concurrency groups to GitHub's CI/CD workflows, preventing
  multiple workflows from the same PR to be launched concurrently.
  (SmartSim-PR439_)
- Torch changed their preferred indexing when trying to install
  their provided wheels. Updated the `pip install` command within
  `smart build` to ensure that the appropriate packages can be found.
  (SmartSim-PR449_)


.. _SmartSim-PR485: https://github.com/CrayLabs/SmartSim/pull/485
.. _SmartSim-PR482: https://github.com/CrayLabs/SmartSim/pull/482
.. _SmartSim-PR479: https://github.com/CrayLabs/SmartSim/pull/479
.. _SmartSim-PR477: https://github.com/CrayLabs/SmartSim/pull/477
.. _SmartSim-PR476: https://github.com/CrayLabs/SmartSim/pull/476
.. _SmartSim-PR475: https://github.com/CrayLabs/SmartSim/pull/475
.. _SmartSim-PR473: https://github.com/CrayLabs/SmartSim/pull/473
.. _SmartSim-PR472: https://github.com/CrayLabs/SmartSim/pull/472
.. _SmartSim-PR471: https://github.com/CrayLabs/SmartSim/pull/471
.. _SmartSim-PR468: https://github.com/CrayLabs/SmartSim/pull/468
.. _SmartSim-PR467: https://github.com/CrayLabs/SmartSim/pull/467
.. _SmartSim-PR465: https://github.com/CrayLabs/SmartSim/pull/465
.. _SmartSim-PR464: https://github.com/CrayLabs/SmartSim/pull/464
.. _SmartSim-PR462: https://github.com/CrayLabs/SmartSim/pull/462
.. _SmartSim-PR461: https://github.com/CrayLabs/SmartSim/pull/461
.. _SmartSim-PR459: https://github.com/CrayLabs/SmartSim/pull/459
.. _SmartSim-PR458: https://github.com/CrayLabs/SmartSim/pull/458
.. _SmartSim-PR453: https://github.com/CrayLabs/SmartSim/pull/453
.. _SmartSim-PR452: https://github.com/CrayLabs/SmartSim/pull/452
.. _SmartSim-PR451: https://github.com/CrayLabs/SmartSim/pull/451
.. _SmartSim-PR449: https://github.com/CrayLabs/SmartSim/pull/449
.. _SmartSim-PR448: https://github.com/CrayLabs/SmartSim/pull/448
.. _SmartSim-PR446: https://github.com/CrayLabs/SmartSim/pull/446
.. _SmartSim-PR439: https://github.com/CrayLabs/SmartSim/pull/439

0.6.0
-----

Released on 18 December, 2023

Description

- Conflicting directives in the SmartSim packaging instructions were fixed
- `sacct` and `sstat` errors are now fatal for Slurm-based workflow executions
- Added documentation section about ML features and TorchScript
- Added TorchScript functions to Online Analysis tutorial
- Added multi-DB example to documentation
- Improved test stability on HPC systems
- Added support for producing & consuming telemetry outputs
- Split tests into groups for parallel execution in CI/CD pipeline
- Change signature of `Experiment.summary()`
- Expose first_device parameter for scripts, functions, models
- Added support for MINBATCHTIMEOUT in model execution
- Remove support for RedisAI 1.2.5, use RedisAI 1.2.7 commit
- Add support for multiple databases

Detailed Notes

- Several conflicting directives between the `setup.py` and the `setup.cfg` were fixed
  to mitigate warnings issued when building the pip wheel. (SmartSim-PR435_)
- When the Slurm functions `sacct` and `sstat` returned an error, it would be ignored
  and SmartSim's state could become inconsistent. To prevent this, errors
  raised by `sacct` or `sstat` now result in an exception. (SmartSim-PR392_)
- A section named *ML Features* was added to documentation. It contains multiple
  examples of how ML models and functions can be added to and executed on the DB.
  TorchScript-based post-processing was added to the *Online Analysis* tutorial (SmartSim-PR411_)
- An example of how to use multiple Orchestrators concurrently was added to the documentation (SmartSim-PR409_)
- The test infrastructure was improved. Tests on HPC system are now stable, and issues such
  as non-stopped `Orchestrators` or experiments created in the wrong paths have been fixed (SmartSim-PR381_)
- A telemetry monitor was added to check updates and produce events for SmartDashboard (SmartSim-PR426_)
- Split tests into `group_a`, `group_b`, `slow_tests` for parallel execution in CI/CD pipeline (SmartSim-PR417_, SmartSim-PR424_)
- Change `format` argument to `style` in `Experiment.summary()`, this is
  an API break (SmartSim-PR391_)
- Added support for first_device parameter for scripts, functions,
  and models. This causes them to be loaded to the first num_devices
  beginning with first_device (SmartSim-PR394_)
- Added support for MINBATCHTIMEOUT in model execution, which caps the delay
  waiting for a minimium number of model execution operations to accumulate
  before executing them as a batch (SmartSim-PR387_)
- RedisAI 1.2.5 is not supported anymore. The only RedisAI version
  is now 1.2.7. Since the officially released RedisAI 1.2.7 has a
  bug which breaks the build process on Mac OSX, it was decided to
  use commit 634916c_ from RedisAI's GitHub repository, where such
  bug has been fixed. This applies to all operating systems. (SmartSim-PR383_)
- Add support for creation of multiple databases with unique identifiers. (SmartSim-PR342_)


.. _SmartSim-PR435: https://github.com/CrayLabs/SmartSim/pull/435
.. _SmartSim-PR392: https://github.com/CrayLabs/SmartSim/pull/392
.. _SmartSim-PR411: https://github.com/CrayLabs/SmartSim/pull/411
.. _SmartSim-PR409: https://github.com/CrayLabs/SmartSim/pull/409
.. _SmartSim-PR381: https://github.com/CrayLabs/SmartSim/pull/381
.. _SmartSim-PR426: https://github.com/CrayLabs/SmartSim/pull/426
.. _SmartSim-PR424: https://github.com/CrayLabs/SmartSim/pull/424
.. _SmartSim-PR417: https://github.com/CrayLabs/SmartSim/pull/417
.. _SmartSim-PR391: https://github.com/CrayLabs/SmartSim/pull/391
.. _SmartSim-PR342: https://github.com/CrayLabs/SmartSim/pull/342
.. _SmartSim-PR394: https://github.com/CrayLabs/SmartSim/pull/394
.. _SmartSim-PR387: https://github.com/CrayLabs/SmartSim/pull/387
.. _SmartSim-PR383: https://github.com/CrayLabs/SmartSim/pull/383
.. _634916c: https://github.com/RedisAI/RedisAI/commit/634916c722e718cc6ea3fad46e63f7d798f9adc2
.. _SmartSim-PR342: https://github.com/CrayLabs/SmartSim/pull/342


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

- Add methods to allow users to inspect files attached to models and ensembles. (SmartSim-PR352_)
- Add a `smart info` target to provide rudimentary information about the SmartSim installation. (SmartSim-PR350_)
- Remove unnecessary generation producing unexpected directories in the test suite. (SmartSim-PR349_)
- Add support for heterogeneous jobs to `SrunSettings` by allowing users to set the `--het-group` parameter. (SmartSim-PR346_)
- Provide clearer guidelines on how to contribute to SmartSim. (SmartSim-PR344_)
- Integrate `PalsMpiexecSettings` into the `Experiment` factory methods when using the `"pals"` launcher. (SmartSim-PR343_)
- Create public properties where appropriate to mitigate `protected-access` errors. (SmartSim-PR341_)
- Fix a failure to execute `_prep_colocated_db` due to incorrect named attr check. (SmartSim-PR339_)
- Enabled and mitigated mypy `disallow_any_generics` and `warn_return_any`. (SmartSim-PR338_)
- Add a `smart validate` target to provide a simple smoke test to assess a SmartSim build. (SmartSim-PR336_, SmartSim-PR351_)
- Add typehints to `smartsim._core.launcher.step.*`. (SmartSim-PR334_)
- Log errors reported from slurm WLM when attempts to retrieve status fail. (SmartSim-PR331_, SmartSim-PR332_)
- Fix incorrectly formatted positional arguments in log format strings. (SmartSim-PR330_)
- Ensure that launchers pass environment variables to unmanaged job steps. (SmartSim-PR329_)
- Add additional tests surrounding the `RAI_PATH` configuration environment variable. (SmartSim-PR328_)
- Remove unnecessary execution of unescaped shell commands. (SmartSim-PR327_)
- Add error if user calls get_allocation with reserved keywords in slurm get_allocation. (SmartSim-PR325_)
- Add error when user requests CPU with devices greater than 1 within add_ml_model and add_script. (SmartSim-PR324_)
- Update documentation surrounding ensemble key prefixing. (SmartSim-PR322_)
- Fix formatting of the Frontier site installation. (SmartSim-PR321_)
- Update pylint dependency, update .pylintrc, mitigate non-breaking issues, suppress api breaks. (SmartSim-PR311_)
- Refactor the `smart` CLI to use subparsers for better documentation and extension. (SmartSim-PR308_)

.. _SmartSim-PR352: https://github.com/CrayLabs/SmartSim/pull/352
.. _SmartSim-PR351: https://github.com/CrayLabs/SmartSim/pull/351
.. _SmartSim-PR350: https://github.com/CrayLabs/SmartSim/pull/350
.. _SmartSim-PR349: https://github.com/CrayLabs/SmartSim/pull/349
.. _SmartSim-PR346: https://github.com/CrayLabs/SmartSim/pull/346
.. _SmartSim-PR344: https://github.com/CrayLabs/SmartSim/pull/344
.. _SmartSim-PR343: https://github.com/CrayLabs/SmartSim/pull/343
.. _SmartSim-PR341: https://github.com/CrayLabs/SmartSim/pull/341
.. _SmartSim-PR339: https://github.com/CrayLabs/SmartSim/pull/339
.. _SmartSim-PR338: https://github.com/CrayLabs/SmartSim/pull/338
.. _SmartSim-PR336: https://github.com/CrayLabs/SmartSim/pull/336
.. _SmartSim-PR334: https://github.com/CrayLabs/SmartSim/pull/334
.. _SmartSim-PR332: https://github.com/CrayLabs/SmartSim/pull/332
.. _SmartSim-PR331: https://github.com/CrayLabs/SmartSim/pull/331
.. _SmartSim-PR330: https://github.com/CrayLabs/SmartSim/pull/330
.. _SmartSim-PR329: https://github.com/CrayLabs/SmartSim/pull/329
.. _SmartSim-PR328: https://github.com/CrayLabs/SmartSim/pull/328
.. _SmartSim-PR327: https://github.com/CrayLabs/SmartSim/pull/327
.. _SmartSim-PR325: https://github.com/CrayLabs/SmartSim/pull/325
.. _SmartSim-PR324: https://github.com/CrayLabs/SmartSim/pull/324
.. _SmartSim-PR322: https://github.com/CrayLabs/SmartSim/pull/322
.. _SmartSim-PR321: https://github.com/CrayLabs/SmartSim/pull/321
.. _SmartSim-PR311: https://github.com/CrayLabs/SmartSim/pull/311
.. _SmartSim-PR308: https://github.com/CrayLabs/SmartSim/pull/308


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

- Updates SmartRedis to the most current release (SmartSim-PR316_)
- Fixes and enhancements to documentation (SmartSim-PR317_, SmartSim-PR314_, SmartSim-PR287_)
- Various fixes and enhancements to the test suite (SmartSim-PR315_, SmartSim-PR312_, SmartSim-PR310_, SmartSim-PR302_, SmartSim-PR283_)
- Fix a defect in the tests related to database models and scripts that was
  causing key collisions when testing on workload managers (SmartSim-PR313_)
- Remove `requirements.txt` and other places where dependencies were defined. (SmartSim-PR307_)
- Fix defect where dictionaries used to create run settings can be changed
  unexpectedly due to copy-by-ref (SmartSim-PR305_)
- The underlying code for Model.add_ml_model() and Model.add_script() was fixed
  to correctly handle multi-GPU configurations.  Tests were updated to run on
  non-local launchers.  Documentation was updated and fixed.  Also, the default
  testing interface has been changed to lo instead of ipogif. (SmartSim-PR304_)
- Typehints have been added. A makefile target `make check-mypy` executes static
  analysis with mypy. (SmartSim-PR295_, SmartSim-PR301_, SmartSim-PR303_)
- Replace `limit_app_cpus` with `limit_db_cpus` for co-located orchestrators.
  This resolves some incorrect behavior/assumptions about how the application
  would be pinned.  Instead, users should directly specify the binding options in
  their application using the options appropriate for their launcher (SmartSim-PR306_)
- Simplify code in `random_permutations` parameter generation strategy (SmartSim-PR300_)
- Remove wait time associated with Experiment launch summary (SmartSim-PR298_)
- Update Redis conf file to conform with Redis v7.0.5 conf file (SmartSim-PR293_)
- Migrate from redis-py-cluster to redis-py for cluster status checks (SmartSim-PR292_)
- Update full test suite to no longer require a tensorflow wheel to be available at test time. (SmartSim-PR291_)
- Correct spelling of colocated in doc strings (SmartSim-PR290_)
- Deprecated launcher-specific orchestrators, constants, and ML
  utilities were removed. (SmartSim-PR289_)
- Relax the coloredlogs version to be greater than 10.0 (SmartSim-PR288_)
- Update the Github Actions runner image from `macos-10.15`` to `macos-12``. The
  former began deprecation in May 2022 and was finally removed in May 2023. (SmartSim-PR285_)
- The Fortran tutorials had not been fully updated to show how to handle
  return/error codes. These have now all been updated. (SmartSim-PR284_)
- Orchestrator and Colocated DB now accept a list of interfaces to bind to. The
  argument name is still `interface` for backward compatibility reasons. (SmartSim-PR281_)
- Typehints have been added to public APIs. A makefile target to execute static
  analysis with mypy is available `make check-mypy`. (SmartSim-PR295_)

.. _SmartSim-PR317: https://github.com/CrayLabs/SmartSim/pull/317
.. _SmartSim-PR316: https://github.com/CrayLabs/SmartSim/pull/316
.. _SmartSim-PR315: https://github.com/CrayLabs/SmartSim/pull/314
.. _SmartSim-PR314: https://github.com/CrayLabs/SmartSim/pull/314
.. _SmartSim-PR313: https://github.com/CrayLabs/SmartSim/pull/313
.. _SmartSim-PR312: https://github.com/CrayLabs/SmartSim/pull/312
.. _SmartSim-PR310: https://github.com/CrayLabs/SmartSim/pull/310
.. _SmartSim-PR307: https://github.com/CrayLabs/SmartSim/pull/307
.. _SmartSim-PR306: https://github.com/CrayLabs/SmartSim/pull/306
.. _SmartSim-PR305: https://github.com/CrayLabs/SmartSim/pull/305
.. _SmartSim-PR304: https://github.com/CrayLabs/SmartSim/pull/304
.. _SmartSim-PR303: https://github.com/CrayLabs/SmartSim/pull/303
.. _SmartSim-PR302: https://github.com/CrayLabs/SmartSim/pull/302
.. _SmartSim-PR301: https://github.com/CrayLabs/SmartSim/pull/301
.. _SmartSim-PR300: https://github.com/CrayLabs/SmartSim/pull/300
.. _SmartSim-PR298: https://github.com/CrayLabs/SmartSim/pull/298
.. _SmartSim-PR295: https://github.com/CrayLabs/SmartSim/pull/295
.. _SmartSim-PR293: https://github.com/CrayLabs/SmartSim/pull/293
.. _SmartSim-PR292: https://github.com/CrayLabs/SmartSim/pull/292
.. _SmartSim-PR291: https://github.com/CrayLabs/SmartSim/pull/291
.. _SmartSim-PR290: https://github.com/CrayLabs/SmartSim/pull/290
.. _SmartSim-PR289: https://github.com/CrayLabs/SmartSim/pull/289
.. _SmartSim-PR288: https://github.com/CrayLabs/SmartSim/pull/288
.. _SmartSim-PR287: https://github.com/CrayLabs/SmartSim/pull/287
.. _SmartSim-PR285: https://github.com/CrayLabs/SmartSim/pull/285
.. _SmartSim-PR284: https://github.com/CrayLabs/SmartSim/pull/284
.. _SmartSim-PR283: https://github.com/CrayLabs/SmartSim/pull/283
.. _SmartSim-PR281: https://github.com/CrayLabs/SmartSim/pull/281

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
  variable will not be assigned the desired value with `--export`. (SmartSim-PR270_)
- The PyTorch and TensorFlow data loaders were update to make use of aggregation lists. This breaks their API, but makes them easier to use. (SmartSim-PR264_)
- The support for Ray was dropped, as its most recent versions caused problems when deployed through SmartSim.
  We plan to release a separate add-on library to accomplish the same results. If
  you are interested in getting the Ray launch functionality back in your workflow, please get in touch with us! (SmartSim-PR263_)
- Update from Redis version 6.0.8 to 7.0.5. (SmartSim-PR258_)
- Adds support for Python 3.10 without the ONNX machine learning backend. Deprecates support for
  Python 3.7 as it will stop receiving security updates. Deprecates support for RedisAI 1.2.3.
  Update the build process to be able to correctly fetch supported dependencies. If a user
  attempts to build an unsupported dependency, an error message is shown highlighting the
  discrepancy. (SmartSim-PR256_)
- Models were given a `batch_settings` attribute. When launching a model through `Experiment.start`
  the `Experiment` will first check for a non-nullish value at that attribute. If the check is
  satisfied, the `Experiment` will attempt to wrap the underlying run command in a batch job using
  the object referenced at `Model.batch_settings` as the batch settings for the job. If the check
  is not satisfied, the `Model` is launched in the traditional manner as a job step. (SmartSim-PR245_)
- Fix bug in colocated database entrypoint stemming from uninitialized variables. This bug affects PyTorch models being loaded into the database. (SmartSim-PR237_)
- The release of RedisAI 1.2.7 allows us to update support for recent versions of PyTorch, Tensorflow, and ONNX (SmartSim-PR234_)
- Make installation of correct Torch backend more reliable according to instruction from PyTorch
- In addition to TCP, add UDS support for colocating an orchestrator with models. Methods
  `Model.colocate_db_tcp` and `Model.colocate_db_uds` were added to expose this functionality.
  The `Model.colocate_db` method remains and uses TCP for backward compatibility (SmartSim-PR246_)

.. _SmartSim-PR270: https://github.com/CrayLabs/SmartSim/pull/270
.. _SmartSim-PR264: https://github.com/CrayLabs/SmartSim/pull/264
.. _SmartSim-PR263: https://github.com/CrayLabs/SmartSim/pull/263
.. _SmartSim-PR258: https://github.com/CrayLabs/SmartSim/pull/258
.. _SmartSim-PR256: https://github.com/CrayLabs/SmartSim/pull/256
.. _SmartSim-PR246: https://github.com/CrayLabs/SmartSim/pull/246
.. _SmartSim-PR245: https://github.com/CrayLabs/SmartSim/pull/245
.. _SmartSim-PR237: https://github.com/CrayLabs/SmartSim/pull/237
.. _SmartSim-PR234: https://github.com/CrayLabs/SmartSim/pull/234


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

 - Add ability to use base ``RunSettings`` on a Slurm, or PBS launchers (SmartSim-PR90_)
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

.. _sr_changelog:

SmartRedis
==========

.. include:: ../smartredis/doc/changelog.rst
    :start-line: 3
