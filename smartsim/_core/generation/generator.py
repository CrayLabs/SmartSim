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
import pickle
import shutil
import typing as t
from datetime import datetime
from glob import glob
from logging import DEBUG, INFO
from os import mkdir, path, symlink
from os.path import join, relpath
import subprocess
import sys


from tabulate import tabulate

from ...entity import Application, TaggedFilesHierarchy
from ...entity.files import EntityFiles
from ...launchable import Job
from ...log import get_logger
from ..entrypoints import file_operations
from ..entrypoints.file_operations import get_parser

logger = get_logger(__name__)
logger.propagate = False


class Generator:
    """The primary job of the generator is to create the file structure
    for a SmartSim Experiment. The Generator is also responsible for
    writing files into a Job directory.
    """

    def __init__(self, exp_path: str, run_id: str) -> None:
        """Initialize a generator object

        The Generator class is responsible for creating Job directories.
        It ensures that paths adhere to SmartSim path standards. Additionally,
        it creates a log directory for telemetry data to handle symlinking,
        configuration, and file copying to the job directory.

        :param gen_path: Path in which files need to be generated
        :param run_ID: The id of the Experiment
        :param job: Reference to a name, SmartSimEntity and LaunchSettings
        """
        self.exp_path = pathlib.Path(exp_path)
        """The path under which the experiment operate"""
        self.run_id = run_id
        """The runID for Experiment.start"""


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

    def log_file(self, log_path: str) -> str:
        """Returns the location of the file
        summarizing the parameters used for the last generation
        of all generated entities.

        :returns: path to file with parameter settings
        """
        return join(log_path, "smartsim_params.txt")

    def generate_job(self, job: Job) -> str:
        """Generate the directories

        Generate the file structure for a SmartSim experiment. This
        includes writing and configuring input files for a job.

        To have files or directories present in the created job
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
        job_path = self._generate_job_path(job)
        log_path = self._generate_log_path(job)

        with open(self.log_file(log_path), mode="w", encoding="utf-8") as log_file:
            dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            log_file.write(f"Generation start date and time: {dt_string}\n")

        # Perform file system operations on attached files
        self._build_operations(job, job_path)

        # Return Job directory path
        return job_path


    def _generate_job_path(self, job: Job) -> str:
        """
        Generates the directory path for a job based on its creation type
        (whether created via ensemble or job init).

        :param job: The Job object
        :param gen_path: The base path for job generation
        :param run_ID: The experiments unique run ID
        :returns str: The generated path for the job.
        """
        job_type = f"{job.__class__.__name__.lower()}s"
        job_path = (
            self.exp_path /
            self.run_id /
            job_type /
            job.name /
            "run"
        )
        # Create Job directory
        job_path.mkdir(exist_ok=True, parents=True)
        return job_path


    def _generate_log_path(self, job: Job) -> str:
        """
        Generate the path for the log folder.

        :param gen_path: The base path job generation
        :returns str: The generated path for the log directory
        """
        job_type = f"{job.__class__.__name__.lower()}s"
        log_path = (
            self.exp_path /
            self.run_id /
            job_type /
            job.name /
            "log"
        )
        log_path.mkdir(exist_ok=True, parents=True)
        return log_path

    
    def _build_operations(self, job: Job, job_path: str) -> None:
        """This method generates file system operations based on the provided application.
        It processes three types of operations: to_copy, to_symlink, and to_configure.
        For each type, it calls the corresponding private methods and appends the results
        to the `file_operation_list`.

        :param app: The application for which operations are generated.
        :return: A list of lists containing file system operations.
        """
        app = t.cast(Application, job.entity)
        self._get_symlink_file_system_operation(app, job_path)
        self._write_tagged_entity_files(app, job_path)
        self._get_copy_file_system_operation(app, job_path)

    @staticmethod
    def _get_copy_file_system_operation(app: Application, dest: str) -> None:
        """Get copy file system operation for a file.

        :param app: The Application attached to the Job
        :param dest: Path to copy files
        """
        if app.files is None:
            return
        for src in app.files.copy:
            if os.path.isdir(src):
                subprocess.run(args=[sys.executable, "-m", "smartsim._core.entrypoints.file_operations", "copy", src, dest, "--dirs_exist_ok"])
            else:
                subprocess.run(args=[sys.executable, "-m", "smartsim._core.entrypoints.file_operations", "copy", src, dest])

    @staticmethod
    def _get_symlink_file_system_operation(app: Application, dest: str) -> None:
        """Get symlink file system operation for a file.

        :param app: The Application attached to the Job
        :param dest: Path to symlink files
        """
        if app.files is None:
            return
        parser = get_parser()
        for src in app.files.link:
            # # Normalize the path to remove trailing slashes
            normalized_path = os.path.normpath(src)
            # # Get the parent directory (last folder)
            parent_dir = os.path.basename(normalized_path)
            dest = os.path.join(dest, parent_dir)
            subprocess.run(args=[sys.executable, "-m", "smartsim._core.entrypoints.file_operations", "symlink", src, dest])

    @staticmethod
    def _write_tagged_entity_files(app: Application, dest: str) -> None:
        """Read, configure and write the tagged input files for
           a Application instance within an ensemble. This function
           specifically deals with the tagged files attached to
           an Ensemble.

        :param app: The Application attached to the Job
        :param dest: Path to configure files
        """
        if app.files is None:
            return
        if app.files.tagged:
            to_write = []

            def _build_tagged_files(tagged: TaggedFilesHierarchy) -> None:
                """Using a TaggedFileHierarchy, reproduce the tagged file
                directory structure

                :param tagged: a TaggedFileHierarchy to be built as a
                               directory structure
                """
                for file in tagged.files:
                    dst_path = path.join(dest, tagged.base, path.basename(file))
                    shutil.copyfile(file, dst_path)
                    to_write.append(dst_path)

                for tagged_dir in tagged.dirs:
                    mkdir(path.join(dest, tagged.base, path.basename(tagged_dir.base)))
                    _build_tagged_files(tagged_dir)

            if app.files.tagged_hierarchy:
                _build_tagged_files(app.files.tagged_hierarchy)

            # Pickle the dictionary
            pickled_dict = pickle.dumps(app.params)
            # Default tag delimiter
            tag = ";"
            # Encode the pickled dictionary with Base64
            encoded_dict = base64.b64encode(pickled_dict).decode("ascii")
            parser = get_parser()
            for dest_path in to_write:
                subprocess.run(args=[sys.executable, "-m", "smartsim._core.entrypoints.file_operations", "configure", dest_path, dest_path, tag, encoded_dict])
                # cmd = f"configure {dest_path} {dest_path} {tag} {encoded_dict}"
                # args = cmd.split()
                # ns = parser.parse_args(args)
                # file_operations.configure(ns)

            # TODO address in ticket 723
            # self._log_params(entity, files_to_params)

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
