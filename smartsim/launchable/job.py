# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from smartsim.launchable.basejob import BaseJob


class Job(BaseJob):

    ## JOBS will have deep copies (not references) -- so when you change a job - it doesnt change other things
    ## make sure to call this out in the docstring

    # combination of a single entity and launch settings
    # build out Job with all its parents

    # these are all user level objects - that will be fed into exp.start()

    # UNIT Testing for the Job
    # what does unit testing look like for a bunch of nested classes

    """A Job holds a reference to a SmartSimEntity and associated
    LaunchSettings prior to launch.  It is responsible for turning
    the stored entity and launch settings into commands that can be
    executed by a launcher.
    """

    def __init__(self, entity: SmartSimEntity, launch_settings: LaunchSettings) -> None:

        ## make sure these are all robust proper python classes
        ## add all the dunder methods
        # __string
        # __ represents etc.
        # and all the normal class things
        super().__init__()
        self.entity = entity  # deepcopy(entity)?
        self.launch_settings = launch_settings  # deepcopy(entity)?

    # self.warehouse_runner = JobWarehouseRunner # omit for now

    # make sure everything that is suppose to be a abstract method, property, or static method is tagged appropriatelyt

    @abstractmethod
    def get_launch_steps(self) -> LaunchCommands:
        """Return the launch steps corresponding to the
        internal data.

        # Examples of launch steps might be
        # Application, Slurm
        #   -N 4 -n 80 /path/to/exe -i input_file -v
        # Application, Dragon
        #   JSON of a single entry to launch
        # MLWorker, Dragon (inherently uses colocated)
        #   3 JSON entries for entites to launch embedded in colocated descriptor
        # FeatureStore, Dragon
        #   JSON of one application to run

        """
        return JobWarehouseRunner.run(self)

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...
