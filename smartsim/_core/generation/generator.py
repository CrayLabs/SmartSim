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

import base64
import os
import pathlib
import shutil
import typing as t
from datetime import datetime
from distutils import dir_util  # pylint: disable=deprecated-module
from logging import DEBUG, INFO
from os import mkdir, path, symlink
from os.path import join, relpath

from tabulate import tabulate

from ...database import FeatureStore
from ...entity import Application, TaggedFilesHierarchy
from ...launchable import Job, JobGroup
from ...log import get_logger
from ..utils.helpers import create_short_id_str

logger = get_logger(__name__)
logger.propagate = False


class Generator:
    """The primary job of the generator is to create the file structure
    for a SmartSim experiment. The Generator is responsible for reading
    and writing into configuration files as well.
    """

    def __init__(self, gen_path: str, run_ID: str, job: Job) -> None:
        """Initialize a generator object

        The Generator class is responsible for creating Job directories.
        It ensures that paths adhere to SmartSim path standards. Additionally,
        it creates a log directory for telemetry data and handles symlinking,
        configuration, and file copying within the job directory.

        :param gen_path: Path in which files need to be generated
        :param job: Reference to a SmartSimEntity and LaunchSettings
        """
        self.job = job
        # TODO revisit this check
        if job._ensemble_name is None:
            job_type = f"{job.__class__.__name__.lower()}s"
            entity_type = f"{job.entity.__class__.__name__.lower()}-{create_short_id_str()}"
            self.path = os.path.join(
                gen_path,
                run_ID,
                job_type,
                f"{job.name}-{create_short_id_str()}",
                entity_type,
                "run",
            )
        else:
            job_type = "ensembles"
            entity_type = f"{job.entity.__class__.__name__.lower()}-{create_short_id_str()}"
            self.path = os.path.join(
                gen_path,
                run_ID,
                job_type,
                job._ensemble_name,
                f"{job.name}",
                entity_type,
                "run",
            )

    @property
    def log_level(self) -> int:
        """Determines the log level based on the value of the environment
        variable SMARTSIM_LOG_LEVEL.

        If the environment variable is set to "debug", returns the log level DEBUG.
        Otherwise, returns the default log level INFO.

        :return: Log level (DEBUG or INFO)
        """
        # Get the value of the environment variable SMARTSIM_LOG_LEVEL
        env_log_level = os.getenv("SMARTSIM_LOG_LEVEL")

        # Set the default log level to INFO
        default_log_level = INFO

        if env_log_level == "debug":
            return DEBUG
        else:
            return default_log_level

    @property
    def log_file(self) -> str:
        """Returns the location of the file
        summarizing the parameters used for the last generation
        of all generated entities.

        :returns: path to file with parameter settings
        """
        return join(self.path, "smartsim_params.txt")

    def generate_experiment(self) -> str:
        """Run ensemble and experiment file structure generation

        Generate the file structure for a SmartSim experiment. This
        includes the writing and configuring of input files for a
        application.

        To have files or directories present in the created entity
        directories, such as datasets or input files, call
        ``entity.attach_generator_files`` prior to generation. See
        ``entity.attach_generator_files`` for more information on
        what types of files can be included.

        Tagged application files are read, checked for input variables to
        configure, and written. Input variables to configure are
        specified with a tag within the input file itself.
        The default tag is surronding an input value with semicolons.
        e.g. ``THERMO=;90;``

        """
        pathlib.Path(self.path).mkdir(exist_ok=True, parents=True)

        # The log_file only keeps track of the last generation
        # this is to avoid gigantic files in case the user repeats
        # generation several times. The information is anyhow
        # redundant, as it is also written in each entity's dir
        with open(self.log_file, mode="w", encoding="utf-8") as log_file:
            dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            log_file.write(f"Generation start date and time: {dt_string}\n")

        # TODO currently will always be true until FeatureStore impl
        # if isinstance(Application, type(self.job.entity)):
        #     file_operation_list = self.build_operations()
        #     self.execute_file_operations(file_operation_list)
        return self.path

    def execute_file_operations(
        self, file_ops: t.Sequence[t.Sequence[str]]
    ) -> None: ...

    def build_operations(self) -> t.Sequence[t.Sequence[str]]:
        """This method generates file system operations based on the provided application.
        It processes three types of operations: to_copy, to_symlink, and to_configure.
        For each type, it calls the corresponding private methods and appends the results
        to the `file_operation_list`.

        :param app: The application for which operations are generated.
        :return: A list of lists containing file system operations.
        """
        application_files = self.job.entity.files
        file_operation_list: t.List[t.Sequence[str]] = []
        # Generate copy file system operations
        file_operation_list.extend(
            self._get_copy_file_system_operation(file_copy)
            for file_copy in application_files.copy
        )
        # Generate symlink file system operations
        file_operation_list.extend(
            self._get_symlink_file_system_operation(file_link)
            for file_link in application_files.link
        )
        # Generate configure file system operations
        file_operation_list.extend(
            self._write_tagged_entity_files(file_configure)
            for file_configure in application_files.tagged
        )
        return file_operation_list

    def _write_tagged_entity_files(self, configure_file: str) -> t.Sequence[str]:
        """Read, configure and write the tagged input files for
           a Application instance within an ensemble. This function
           specifically deals with the tagged files attached to
           an Ensemble.

        :param entity: a Application instance
        """
        # if entity.files:
        #     to_write = []

        #     def _build_tagged_files(tagged: TaggedFilesHierarchy) -> None:
        #         """Using a TaggedFileHierarchy, reproduce the tagged file
        #         directory structure

        #         :param tagged: a TaggedFileHierarchy to be built as a
        #                        directory structure
        #         """
        #         for file in tagged.files:
        #             dst_path = path.join(entity.path, tagged.base, path.basename(file))
        #             shutil.copyfile(file, dst_path)
        #             to_write.append(dst_path)

        #         for tagged_dir in tagged.dirs:
        #             mkdir(
        #                 path.join(
        #                     entity.path, tagged.base, path.basename(tagged_dir.base)
        #                 )
        #             )
        #             _build_tagged_files(tagged_dir)

        #     if entity.files.tagged_hierarchy:
        #         _build_tagged_files(entity.files.tagged_hierarchy)

        #     # write in changes to configurations
        #     if isinstance(entity, Application):
        #         files_to_params = self._writer.configure_tagged_application_files(
        #             to_write, entity.params
        #         )
        #         self._log_params(entity, files_to_params)
        return ["temporary", "config"]

    # TODO replace with entrypoint operation
    @staticmethod
    def _get_copy_file_system_operation(copy_file: str) -> t.Sequence[str]:
        """Get copy file system operation for a file.

        :param linked_file: The file to be copied.
        :return: A list of copy file system operations.
        """
        return ["temporary", "copy"]

    # TODO replace with entrypoint operation
    @staticmethod
    def _get_symlink_file_system_operation(linked_file: str) -> t.Sequence[str]:
        """Get symlink file system operation for a file.

        :param linked_file: The file to be symlinked.
        :return: A list of symlink file system operations.
        """
        return ["temporary", "link"]

    # TODO to be refactored in ticket 723
    # def _log_params(
    #     self, entity: Application, files_to_params: t.Dict[str, t.Dict[str, str]]
    # ) -> None:
    #     """Log which files were modified during generation

    #     and what values were set to the parameters

    #     :param entity: the application being generated
    #     :param files_to_params: a dict connecting each file to its parameter settings
    #     """
    #     used_params: t.Dict[str, str] = {}
    #     file_to_tables: t.Dict[str, str] = {}
    #     for file, params in files_to_params.items():
    #         used_params.update(params)
    #         table = tabulate(params.items(), headers=["Name", "Value"])
    #         file_to_tables[relpath(file, self.gen_path)] = table

    #     if used_params:
    #         used_params_str = ", ".join(
    #             [f"{name}={value}" for name, value in used_params.items()]
    #         )
    #         logger.log(
    #             level=self.log_level,
    #             msg=f"Configured application {entity.name} with params {used_params_str}",
    #         )
    #         file_table = tabulate(
    #             file_to_tables.items(),
    #             headers=["File name", "Parameters"],
    #         )
    #         log_entry = f"Application name: {entity.name}\n{file_table}\n\n"
    #         with open(self.log_file, mode="a", encoding="utf-8") as logfile:
    #             logfile.write(log_entry)
    #         with open(
    #             join(entity.path, "smartsim_params.txt"), mode="w", encoding="utf-8"
    #         ) as local_logfile:
    #             local_logfile.write(log_entry)

    #     else:
    #         logger.log(
    #             level=self.log_level,
    #             msg=f"Configured application {entity.name} with no parameters",
    #         )
