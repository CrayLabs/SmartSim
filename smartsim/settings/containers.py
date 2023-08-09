# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
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

import shutil
import typing as t

from ..log import get_logger

logger = get_logger(__name__)


class Container:
    """Base class for container types in SmartSim.

    Container types are used to embed all the information needed to
    launch a workload within a container into a single object.

    :param image: local or remote path to container image
    :type image: str
    :param args: arguments to container command
    :type args: str | list[str], optional
    :param mount: paths to mount (bind) from host machine into image.
    :type mount: str | list[str] | dict[str, str], optional
    :param working_directory: path of the working directory within the container
    :type working_directory: str
    """

    def __init__(
        self, image: str, args: str = "", mount: str = "", working_directory: str = ""
    ) -> None:
        # Validate types
        if not isinstance(image, str):
            raise TypeError("image must be a str")
        if not isinstance(args, (str, list)):
            raise TypeError("args must be a str | list")
        if not isinstance(mount, (str, list, dict)):
            raise TypeError("mount must be a str | list | dict")
        if not isinstance(working_directory, str):
            raise TypeError("working_directory must be a str")

        self.image = image
        self.args = args
        self.mount = mount
        self.working_directory = working_directory

    def _containerized_run_command(self, run_command: str) -> str:
        """Return modified run_command with container commands prepended.

        :param run_command: run command from a RunSettings class
        :type run_command: str
        """
        raise NotImplementedError(
            "Containerized run command specification not implemented for this "
            f"Container type: {type(self)}"
        )


class Singularity(Container):
    # pylint: disable=abstract-method
    # todo: determine if _containerized_run_command should be abstract

    """Singularity (apptainer) container type. To be passed into a
    ``RunSettings`` class initializer or ``Experiment.create_run_settings``.

    .. note::

        Singularity integration is currently tested with
        `Apptainer 1.0 <https://apptainer.org/docs/user/1.0/index.html>`_
        with slurm and PBS workload managers only.

        Also, note that user-defined bind paths (``mount`` argument) may be
        disabled by a
        `system administrator
        <https://apptainer.org/docs/admin/1.0/configfiles.html#bind-mount-management>`_


    :param image: local or remote path to container image,
        e.g. ``docker://sylabsio/lolcow``
    :type image: str
    :param args: arguments to 'singularity exec' command
    :type args: str | list[str], optional
    :param mount: paths to mount (bind) from host machine into image.
    :type mount: str | list[str] | dict[str, str], optional
    """

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        super().__init__(*args, **kwargs)

    def _container_cmds(self, default_working_directory: str = "") -> t.List[str]:
        """Return list of container commands to be inserted before exe.
            Container members are validated during this call.

        :raises TypeError: if object members are invalid types
        """
        serialized_args = ""
        if self.args:
            # Serialize args into a str
            if isinstance(self.args, str):
                serialized_args = self.args
            elif isinstance(self.args, list):
                serialized_args = " ".join(self.args)
            else:
                raise TypeError("self.args must be a str | list")

        serialized_mount = ""
        if self.mount:
            if isinstance(self.mount, str):
                serialized_mount = self.mount
            elif isinstance(self.mount, list):
                serialized_mount = ",".join(self.mount)
            elif isinstance(self.mount, dict):
                paths = []
                for host_path, img_path in self.mount.items():
                    if img_path:
                        paths.append(f"{host_path}:{img_path}")
                    else:
                        paths.append(host_path)
                serialized_mount = ",".join(paths)
            else:
                raise TypeError("self.mount must be str | list | dict")

        working_directory = default_working_directory
        if self.working_directory:
            working_directory = self.working_directory

        if working_directory not in serialized_mount:
            if serialized_mount:
                serialized_mount = ",".join([working_directory, serialized_mount])
            else:
                serialized_mount = working_directory
            logger.warning(
                f"Working directory not specified in mount: \n {working_directory}\n"
                "Automatically adding it to the list of bind points"
            )

        # Find full path to singularity
        singularity = shutil.which("singularity")

        # Some systems have singularity available on compute nodes only,
        #   so warn instead of error
        if not singularity:
            logger.warning(
                "Unable to find singularity. Continuing in case singularity is "
                "available on compute node"
            )

        # Construct containerized launch command
        cmd_list = [singularity or "singularity", "exec"]
        if working_directory:
            cmd_list.extend(["--pwd", working_directory])

        if serialized_args:
            cmd_list.append(serialized_args)
        if serialized_mount:
            cmd_list.extend(["--bind", serialized_mount])
        cmd_list.append(self.image)

        return cmd_list
