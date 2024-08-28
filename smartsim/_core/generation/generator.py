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

    def __init__(self, root: pathlib.Path) -> None:
        """Initialize a Generator object

        The class handles symlinking, copying, and configuration of files
        associated with a Jobs entity. Additionally, it writes entity parameters
        used for the specific run into the "smartsim_params.txt" settings file within
        the Jobs log folder.
        """
        self.root = root
        """The root path under which to generate files"""

    def _generate_job_root(self, job: Job, job_index: int) -> pathlib.Path:
        """Generates the root directory for a specific job instance.

        :param job: The Job instance for which the root directory is generated.
        :param job_index: The index of the Job instance (used for naming).
        :returns: The path to the root directory for the Job instance.
        """
        job_type = f"{job.__class__.__name__.lower()}s"
        job_path = self.root / f"{job_type}/{job.name}-{job_index}"
        return pathlib.Path(job_path)

    def _generate_run_path(self, job: Job, job_index: int) -> pathlib.Path:
        """Generates the path for the "run" directory within the root directory
        of a specific Job instance.

        :param job (Job): The Job instance for which the path is generated.
        :param job_index (int): The index of the Job instance (used for naming).
        :returns: The path to the "run" directory for the Job instance.
        """
        path = self._generate_job_root(job, job_index) / "run"
        path.mkdir(exist_ok=False, parents=True)
        return pathlib.Path(path)

    def _generate_log_path(self, job: Job, job_index: int) -> pathlib.Path:
        """
        Generates the path for the "log" directory within the root directory of a specific Job instance.

        :param job: The Job instance for which the path is generated.
        :param job_index: The index of the Job instance (used for naming).
        :returns: The path to the "log" directory for the Job instance.
        """
        path = self._generate_job_root(job, job_index) / "log"
        path.mkdir(exist_ok=False, parents=True)
        return pathlib.Path(path)

    @staticmethod
    def _log_file(log_path: pathlib.Path) -> pathlib.Path:
        """Returns the location of the file
        summarizing the parameters used for the generation
        of the entity.

        :param log_path: Path to log directory
        :returns: Path to file with parameter settings
        """
        return pathlib.Path(log_path) / "smartsim_params.txt"

    @staticmethod
    def _output_files(
        log_path: pathlib.Path, job_name: str
    ) -> t.Tuple[pathlib.Path, pathlib.Path]:
        out_file_path = log_path / f"{job_name}.out"
        err_file_path = log_path / f"{job_name}.err"
        return out_file_path, err_file_path

    def generate_job(
        self, job: Job, job_index: int
    ) -> t.Tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
        """Write and configure input files for a Job.

        To have files or directories present in the created Job
        directory, such as datasets or input files, call
        ``entity.attach_generator_files`` prior to generation.

        Tagged application files are read, checked for input variables to
        configure, and written. Input variables to configure are
        specified with a tag within the input file itself.
        The default tag is surrounding an input value with semicolons.
        e.g. ``THERMO=;90;``

        :param job: The job instance to write and configure files for.
        :param job_path: The path to the "run" directory for the job instance.
        :param log_path: The path to the "log" directory for the job instance.
        """

        # Generate ../job_name/run directory
        job_path = self._generate_run_path(job, job_index)
        # Generate ../job_name/log directory
        log_path = self._generate_log_path(job, job_index)

        # Create and write to the parameter settings file
        with open(self._log_file(log_path), mode="w", encoding="utf-8") as log_file:
            dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            log_file.write(f"Generation start date and time: {dt_string}\n")

        # Create output files
        out_file, err_file = self._output_files(log_path, job.entity.name)

        # Perform file system operations on attached files
        self._build_operations(job, job_path)

        return job_path, out_file, err_file

    @classmethod
    def _build_operations(cls, job: Job, job_path: pathlib.Path) -> None:
        """This method orchestrates file system ops for the attached SmartSim entity.
        It processes three types of file system operations: to_copy, to_symlink, and to_configure.
        For each type, it calls the corresponding private methods that open a subprocess
        to complete each task.

        :param job: The Job to perform file ops on attached entity files
        :param job_path: Path to the Jobs run directory
        """
        app = t.cast(Application, job.entity)
        cls._copy_files(app.files, job_path)
        cls._symlink_files(app.files, job_path)
        cls._write_tagged_files(app.files, app.params, job_path)

    @staticmethod
    def _copy_files(files: t.Union[EntityFiles, None], dest: pathlib.Path) -> None:
        """Perform copy file sys operations on a list of files.

        :param app: The Application attached to the Job
        :param dest: Path to the Jobs run directory
        """
        # Return if no files are attached
        if files is None:
            return
        for src in files.copy:
            if os.path.isdir(src):
                # Remove basename of source
                base_source_name = os.path.basename(src)
                # Attach source basename to destination
                new_dst_path = os.path.join(dest, base_source_name)
                # Copy source contents to new destination path
                subprocess.run(
                    args=[
                        sys.executable,
                        "-m",
                        "smartsim._core.entrypoints.file_operations",
                        "copy",
                        src,
                        new_dst_path,
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
    def _symlink_files(files: t.Union[EntityFiles, None], dest: pathlib.Path) -> None:
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
    def _write_tagged_files(
        files: t.Union[EntityFiles, None],
        params: t.Mapping[str, str],
        dest: pathlib.Path,
    ) -> None:
        """Read, configure and write the tagged input files for
           a Job instance. This function specifically deals with the tagged
           files attached to an entity.

        :param app: The Application attached to the Job
        :param dest: Path to the Jobs run directory
        """
        # Return if no files are attached
        if files is None:
            return
        if files.tagged:
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

            if files.tagged_hierarchy:
                _build_tagged_files(files.tagged_hierarchy)

            # Pickle the dictionary
            pickled_dict = pickle.dumps(params)
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
