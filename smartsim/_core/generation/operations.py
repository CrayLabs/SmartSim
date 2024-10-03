import os
import pathlib
import sys
import typing as t
import pickle
import base64
from dataclasses import dataclass, field

from ..commands import Command

entry_point_path = "smartsim._core.entrypoints.file_operations"
"""Path to file operations module."""

copy_cmd = "copy"
symlink_cmd = "symlink"
configure_cmd = "configure"


def create_final_dest(
    job_root_path: pathlib.Path, dest: t.Union[pathlib.Path, None]
) -> str:
    """Combine the job root path and destination path. Return as a string for
    entry point consumption.

    :param job_root_path: Job root path
    :param dest: Destination path
    :return: Combined path
    :raises ValueError: An error occurred during path combination
    """
    if dest is not None and not isinstance(dest, pathlib.Path):
        raise ValueError(f"Must be absolute path")
    if isinstance(dest, pathlib.Path) and not dest.is_absolute():
        raise ValueError("Invalid destination path")
    if isinstance(dest, pathlib.Path) and " " in str(dest):
        raise ValueError("Path contains spaces, which are not allowed")
    if (
        job_root_path is None
        or job_root_path == pathlib.Path("")
        or isinstance(job_root_path, str)
    ):
        raise ValueError(f"Job root path '{job_root_path}' is not a directory.")
    try:
        combined_path = job_root_path
        if dest:
            combined_path = job_root_path / dest
        return str(combined_path)
    except Exception as e:
        raise ValueError(f"Error combining paths: {e}")


class GenerationContext:
    """Context for file system generation operations."""

    def __init__(self, job_root_path: pathlib.Path):
        self.job_root_path = job_root_path
        """The Job root path"""


class GenerationProtocol(t.Protocol):
    """Protocol for Generation Operations."""

    def format(self, context: GenerationContext) -> Command:
        """Return a formatted Command."""


class CopyOperation(GenerationProtocol):
    """Copy Operation"""

    def __init__(
        self, src: pathlib.Path, dest: t.Optional[pathlib.Path] = None
    ) -> None:
        self.src = src
        self.dest = dest

    def format(self, context: GenerationContext) -> Command:
        """Create Command to invoke copy fs entry point"""
        final_dest = create_final_dest(context.job_root_path, self.dest)
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

    def __init__(self, src: pathlib.Path, dest: t.Optional[pathlib.Path] = None) -> None:
        self.src = src
        self.dest = dest

    def format(self, context: GenerationContext) -> Command:
        """Create Command to invoke symlink fs entry point"""
        normalized_path = os.path.normpath(self.src)
        # # Get the parent directory (last folder)
        parent_dir = os.path.basename(normalized_path)
        final_dest = create_final_dest(context.job_root_path, self.dest)
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
        file_parameters: t.Mapping[str,str],
        dest: t.Optional[pathlib.Path] = None,
        tag: t.Optional[str] = None,
    ) -> None:
        self.src = src
        self.dest = dest
        pickled_dict = pickle.dumps(file_parameters)
        encoded_dict = base64.b64encode(pickled_dict).decode("ascii")
        self.file_parameters = encoded_dict
        self.tag = tag if tag else ";"

    def format(self, context: GenerationContext) -> Command:
        """Create Command to invoke configure fs entry point"""
        final_dest = create_final_dest(context.job_root_path, self.dest)
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


T = t.TypeVar("T", bound=GenerationProtocol)


@dataclass
class FileSysOperationSet:
    """Dataclass to represent a set of FS Operation Objects"""

    # disallow modification - dunder function (post ticket to reevaluate API objects)
    operations: t.List[GenerationProtocol] = field(default_factory=list)
    """Set of FS Objects that match the GenerationProtocol"""

    def add_copy(
        self, src: pathlib.Path, dest: t.Optional[pathlib.Path] = None
    ) -> None:
        """Add a copy operation to the operations list"""
        self.operations.append(CopyOperation(src, dest))

    def add_symlink(
        self, src: pathlib.Path, dest: t.Optional[pathlib.Path] = None
    ) -> None:
        """Add a symlink operation to the operations list"""
        self.operations.append(SymlinkOperation(src, dest))

    def add_configuration(
        self,
        src: pathlib.Path,
        file_parameters: t.Mapping[str,str],
        dest: t.Optional[pathlib.Path] = None,
        tag: t.Optional[str] = None,
    ) -> None:
        """Add a configure operation to the operations list"""
        self.operations.append(ConfigureOperation(src, file_parameters, dest, tag))

    @property
    def copy_operations(self) -> t.List[CopyOperation]:
        """Property to get the list of copy files."""
        return self._filter(CopyOperation)

    @property
    def symlink_operations(self) -> t.List[SymlinkOperation]:
        """Property to get the list of symlink files."""
        return self._filter(SymlinkOperation)

    @property
    def configure_operations(self) -> t.List[ConfigureOperation]:
        """Property to get the list of configure files."""
        return self._filter(ConfigureOperation)

    def _filter(self, type: t.Type[T]) -> t.List[T]:
        return [x for x in self.operations if isinstance(x, type)]