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
import subprocess
import sys
import typing as t
from datetime import datetime
from os import mkdir, path
from os.path import join

from ...entity import Application, TaggedFilesHierarchy
from ...entity.files import EntityFiles
from ...launchable import Job
from ...log import get_logger

logger = get_logger(__name__)
logger.propagate = False


class Generator:
    """The primary job of the Generator is to create the directory and file structure
    for a SmartSim Job. The Generator is also responsible for writing and configuring
    files into the Job directory.
    """

    def __init__(self, exp_path: str, run_id: str) -> None:
        """Initialize a generator object

        The Generator class is responsible for creating Job directories.
        It ensures that paths adhere to SmartSim path standards. Additionally,
        it creates a run directory to handle symlinking,
        configuration, and file copying to the job directory.

        :param gen_path: Path in which files need to be generated
        :param run_ID: The id of the Experiment
        """
        self.exp_path = pathlib.Path(exp_path)
        """The path under which the experiment operate"""
        self.run_id = run_id
        """The runID for Experiment.start"""

    def log_file(self, log_path: pathlib.Path) -> str:
        """Returns the location of the file
        summarizing the parameters used for the generation
        of the entity.

        :param log_path: Path to log directory
        :returns: Path to file with parameter settings
        """
        return join(log_path, "smartsim_params.txt")

    def generate_job(self, job: Job, job_index: int) -> pathlib.Path:
        """Generate the Job directory

        Generate the file structure for a SmartSim Job. This
        includes writing and configuring input files for the entity.

        To have files or directories present in the created Job
        directories, such as datasets or input files, call
        ``entity.attach_generator_files`` prior to generation.

        Tagged application files are read, checked for input variables to
        configure, and written. Input variables to configure are
        specified with a tag within the input file itself.
        The default tag is surronding an input value with semicolons.
        e.g. ``THERMO=;90;``

        """
        # Generate ../job_name/run directory
        job_path = self._generate_job_path(job, job_index)
        # Generate ../job_name/log directory
        log_path = self._generate_log_path(job, job_index)

        # Create and write to the parameter settings file
        with open(self.log_file(log_path), mode="w", encoding="utf-8") as log_file:
            dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            log_file.write(f"Generation start date and time: {dt_string}\n")

        # Perform file system ops
        self._build_operations(job, job_path)

        # Return Job path
        return job_path

    def _generate_job_path(self, job: Job, job_index: int) -> pathlib.Path:
        """
        Generate the run directory for a Job.

        :param job: The Job to generate a directory
        :returns pathlib.Path:: The generated run path for the Job
        """
        job_type = f"{job.__class__.__name__.lower()}s"
        job_path = (
            self.exp_path / self.run_id / job_type / f"{job.name}-{job_index}" / "run"
        )
        # Create Job directory
        job_path.mkdir(exist_ok=True, parents=True)
        return job_path

    def _generate_log_path(self, job: Job, job_index: int) -> pathlib.Path:
        """
        Generate the log directory for a Job.

        :param job: The Job to generate a directory
        :returns pathlib.Path:: The generated log path for the Job
        """
        job_type = f"{job.__class__.__name__.lower()}s"
        log_path = (
            self.exp_path / self.run_id / job_type / f"{job.name}-{job_index}" / "log"
        )
        log_path.mkdir(exist_ok=True, parents=True)
        return log_path

    def _build_operations(self, job: Job, job_path: pathlib.Path) -> None:
        """This method orchestrates file system ops for the attached SmartSim entity.
        It processes three types of file system ops: to_copy, to_symlink, and to_configure.
        For each type, it calls the corresponding private methods that open a subprocess
        to complete each task.

        :param job: The Job to perform file ops on attached entity files
        :param job_path: Path to the Jobs run directory
        """
        app = t.cast(Application, job.entity)
        self._copy_files(app.files, job_path)
        self._symlink_files(app.files, job_path)
        self._write_tagged_files(app, job_path)

    @staticmethod
    def _copy_files(files: EntityFiles | None, dest: pathlib.Path) -> None:
        """Perform copy file sys operations on a list of files.

        :param app: The Application attached to the Job
        :param dest: Path to the Jobs run directory
        """
        # Return if no files are attached
        if files is None:
            return
        for src in files.copy:
            if os.path.isdir(src):
                subprocess.run(
                    args=[
                        sys.executable,
                        "-m",
                        "smartsim._core.entrypoints.file_operations",
                        "copy",
                        src,
                        dest,
                        "--dirs_exist_ok",
                    ]
                )
            else:
                subprocess.run(
                    args=[
                        sys.executable,
                        "-m",
                        "smartsim._core.entrypoints.file_operations",
                        "copy",
                        src,
                        dest,
                    ]
                )

    @staticmethod
    def _symlink_files(files: EntityFiles | None, dest: pathlib.Path) -> None:
        """Perform symlink file sys operations on a list of files.

        :param app: The Application attached to the Job
        :param dest: Path to the Jobs run directory
        """
        # Return if no files are attached
        if files is None:
            return
        for src in files.link:
            # Normalize the path to remove trailing slashes
            normalized_path = os.path.normpath(src)
            # Get the parent directory (last folder)
            parent_dir = os.path.basename(normalized_path)
            # Create destination
            new_dest = os.path.join(str(dest), parent_dir)
            subprocess.run(
                args=[
                    sys.executable,
                    "-m",
                    "smartsim._core.entrypoints.file_operations",
                    "symlink",
                    src,
                    new_dest,
                ]
            )

    @staticmethod
    def _write_tagged_files(app: Application, dest: pathlib.Path) -> None:
        """Read, configure and write the tagged input files for
           a Job instance. This function specifically deals with the tagged
           files attached to an entity.

        :param app: The Application attached to the Job
        :param dest: Path to the Jobs run directory
        """
        # Return if no files are attached
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
            for dest_path in to_write:
                subprocess.run(
                    args=[
                        sys.executable,
                        "-m",
                        "smartsim._core.entrypoints.file_operations",
                        "configure",
                        dest_path,
                        dest_path,
                        tag,
                        encoded_dict,
                    ]
                )

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
