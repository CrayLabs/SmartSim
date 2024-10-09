import base64
import os
import pathlib
import pickle
import sys
import typing as t
from dataclasses import dataclass, field

from ..commands import Command

# pylint: disable=invalid-name
entry_point_path = "smartsim._core.entrypoints.file_operations"
"""Path to file operations module"""

copy_cmd = "copy"
"""Copy file operations command"""
symlink_cmd = "symlink"
"""Symlink file operations command"""
configure_cmd = "configure"
"""Configure file operations command"""


def _create_final_dest(job_root_path: pathlib.Path, dest: pathlib.Path) -> str:
    """Combine the job root path and destination path. Return as a string for
    entry point consumption.

    :param job_root_path: Job root path
    :param dest: Destination path
    :return: Combined path
    :raises ValueError: An error occurred during path combination
    """
    combined_path = job_root_path / dest
    return str(combined_path)


def _check_src_and_dest_path(
    src: pathlib.Path, dest: t.Union[pathlib.Path, None]
) -> None:
    """Validate that the provided source and destination paths are
    of type pathlib.Path

    :param src: The source path to be checked.
    :param dest: The destination path to be checked.
    :raises TypeError: If either src or dest is not an instance of pathlib.Path
    """
    if not isinstance(src, pathlib.Path) or not src.is_absolute():
        raise TypeError(f"src must be of type pathlib.Path, not {type(src).__name__}")
    if dest is not None and not isinstance(dest, pathlib.Path):
        raise TypeError(
            f"dest must be of type pathlib.Path or None, not {type(dest).__name__}"
        )
    if isinstance(dest, pathlib.Path) and dest.is_absolute():
        raise ValueError("Invalid destination path")


def _check_run_path(run_path: pathlib.Path) -> None:
    """Validate that the provided run path is of type pathlib.Path

    :param run_path: The run path to be checked
    :raises TypeError: If either run path is not an instance of pathlib.Path
    :raises ValueError: If the run path is not a directory
    """
    if not isinstance(run_path, pathlib.Path):
        raise TypeError(
            f"run_path must be of type pathlib.Path, not {type(run_path).__name__}"
        )


class GenerationContext:
    """Context for file system generation operations."""

    def __init__(self, job_root_path: pathlib.Path):
        _check_run_path(job_root_path)
        self.job_root_path = job_root_path
        """The Job run path"""


class GenerationProtocol(t.Protocol):
    """Protocol for Generation Operations."""

    def format(self, context: GenerationContext) -> Command:
        """Return a formatted Command."""


class CopyOperation(GenerationProtocol):
    """Copy Operation"""

    def __init__(
        self, src: pathlib.Path, dest: t.Optional[pathlib.Path] = None
    ) -> None:
        """Initialize a CopyOperation object

        :param src: Path to source
        :param dest: Path to destination
        """
        _check_src_and_dest_path(src, dest)
        self.src = src
        self.dest = dest or pathlib.Path(src.name)

    def format(self, context: GenerationContext) -> Command:
        """Create Command to invoke copy file system entry point

        :param context: Context for copy operation
        :return: Copy Command
        """
        final_dest = _create_final_dest(context.job_root_path, self.dest)
        return Command(
            [
                sys.executable,
                "-m",
                entry_point_path,
                copy_cmd,
                str(self.src),
                final_dest,
                "--dirs_exist_ok",
            ]
        )


class SymlinkOperation(GenerationProtocol):
    """Symlink Operation"""

    def __init__(
        self, src: pathlib.Path, dest: t.Optional[pathlib.Path] = None
    ) -> None:
        """Initialize a SymlinkOperation object

        :param src: Path to source
        :param dest: Path to destination
        """
        _check_src_and_dest_path(src, dest)
        self.src = src
        self.dest = dest or pathlib.Path(src.name)

    def format(self, context: GenerationContext) -> Command:
        """Create Command to invoke symlink file system entry point

        :param context: Context for symlink operation
        :return: Symlink Command
        """
        normalized_path = os.path.normpath(self.src)
        parent_dir = os.path.dirname(normalized_path)
        final_dest = _create_final_dest(context.job_root_path, self.dest)
        new_dest = os.path.join(final_dest, parent_dir)
        return Command(
            [
                sys.executable,
                "-m",
                entry_point_path,
                symlink_cmd,
                str(self.src),
                new_dest,
            ]
        )


class ConfigureOperation(GenerationProtocol):
    """Configure Operation"""

    def __init__(
        self,
        src: pathlib.Path,
        file_parameters: t.Mapping[str, str],
        dest: t.Optional[pathlib.Path] = None,
        tag: t.Optional[str] = None,
    ) -> None:
        """Initialize a ConfigureOperation

        :param src: Path to source
        :param file_parameters: File parameters to find and replace
        :param dest: Path to destination
        :param tag: Tag to use for find and replacement
        """
        _check_src_and_dest_path(src, dest)
        self.src = src
        self.dest = dest or pathlib.Path(src.name)
        pickled_dict = pickle.dumps(file_parameters)
        encoded_dict = base64.b64encode(pickled_dict).decode("ascii")
        self.file_parameters = encoded_dict
        self.tag = tag if tag else ";"

    def format(self, context: GenerationContext) -> Command:
        """Create Command to invoke configure file system entry point

        :param context: Context for configure operation
        :return: Configure Command
        """
        final_dest = _create_final_dest(context.job_root_path, self.dest)
        return Command(
            [
                sys.executable,
                "-m",
                entry_point_path,
                configure_cmd,
                str(self.src),
                final_dest,
                self.tag,
                self.file_parameters,
            ]
        )


GenerationProtocolT = t.TypeVar("GenerationProtocolT", bound=GenerationProtocol)


@dataclass
class FileSysOperationSet:
    """Dataclass to represent a set of file system operation objects"""

    # TODO disallow modification - dunder function (post ticket to reevaluate API objects)
    operations: t.List[GenerationProtocol] = field(default_factory=list)
    """Set of file system objects that match the GenerationProtocol"""

    def add_copy(
        self, src: pathlib.Path, dest: t.Optional[pathlib.Path] = None
    ) -> None:
        """Add a copy operation to the operations list

        :param src: Path to source
        :param dest: Path to destination
        """
        self.operations.append(CopyOperation(src, dest))

    def add_symlink(
        self, src: pathlib.Path, dest: t.Optional[pathlib.Path] = None
    ) -> None:
        """Add a symlink operation to the operations list

        :param src: Path to source
        :param dest: Path to destination
        """
        self.operations.append(SymlinkOperation(src, dest))

    def add_configuration(
        self,
        src: pathlib.Path,
        file_parameters: t.Mapping[str, str],
        dest: t.Optional[pathlib.Path] = None,
        tag: t.Optional[str] = None,
    ) -> None:
        """Add a configure operation to the operations list

        :param src: Path to source
        :param file_parameters: File parameters to find and replace
        :param dest: Path to destination
        :param tag: Tag to use for find and replacement
        """
        self.operations.append(ConfigureOperation(src, file_parameters, dest, tag))

    @property
    def copy_operations(self) -> list[CopyOperation]:
        """Property to get the list of copy files.

        :return: List of CopyOperation objects
        """
        return self._filter(CopyOperation)

    @property
    def symlink_operations(self) -> list[SymlinkOperation]:
        """Property to get the list of symlink files.

        :return: List of SymlinkOperation objects
        """
        return self._filter(SymlinkOperation)

    @property
    def configure_operations(self) -> list[ConfigureOperation]:
        """Property to get the list of configure files.

        :return: List of ConfigureOperation objects
        """
        return self._filter(ConfigureOperation)

    def _filter(self, type_: type[GenerationProtocolT]) -> list[GenerationProtocolT]:
        """Filters the operations list to include only instances of the
        specified type.

        :param type: The type of operations to filter
        :return: A list of operations that are instances of the specified type
        """
        return [x for x in self.operations if isinstance(x, type_)]
