import pathlib
import typing as t
from dataclasses import dataclass, field
from .utils.helpers import check_src_and_dest_path
from .operations import default_tag

# TODO do we need to add check for tags?
# TODO do I need to add checks for file_params?
class EnsembleGenerationProtocol(t.Protocol):
    """Protocol for Ensemble Generation Operations."""
    src: pathlib.Path
    """Path to source"""
    dest: t.Optional[pathlib.Path]
    """Path to destination"""


class EnsembleCopyOperation(EnsembleGenerationProtocol):
    """Ensemble Copy Operation"""

    def __init__(
        self, src: pathlib.Path, dest: t.Optional[pathlib.Path] = None
    ) -> None:
        """Initialize a EnsembleCopyOperation object

        :param src: Path to source
        :param dest: Path to destination
        """
        check_src_and_dest_path(src, dest)
        self.src = src
        """Path to source"""
        self.dest = dest
        """Path to destination"""


class EnsembleSymlinkOperation(EnsembleGenerationProtocol):
    """Ensemble Symlink Operation"""

    def __init__(self, src: pathlib.Path, dest: t.Optional[pathlib.Path] = None) -> None:
        """Initialize a EnsembleSymlinkOperation object

        :param src: Path to source
        :param dest: Path to destination
        """
        check_src_and_dest_path(src, dest)
        self.src = src
        """Path to source"""
        self.dest = dest
        """Path to destination"""


class EnsembleConfigureOperation(EnsembleGenerationProtocol):
    """Ensemble Configure Operation"""

    def __init__(
        self,
        src: pathlib.Path,
        file_parameters:t.Mapping[str,t.Sequence[str]],
        dest: t.Optional[pathlib.Path] = None,
        tag: t.Optional[str] = None,
    ) -> None:
        """Initialize a EnsembleConfigureOperation

        :param src: Path to source
        :param file_parameters: File parameters to find and replace
        :param dest: Path to destination
        :param tag: Tag to use for find and replacement
        """
        check_src_and_dest_path(src, dest)
        self.src = src
        """Path to source"""
        self.dest = dest
        """Path to destination"""
        self.file_parameters = file_parameters
        """File parameters to find and replace"""
        self.tag = tag if tag else default_tag
        """Tag to use for the file"""


EnsembleGenerationProtocolT = t.TypeVar("EnsembleGenerationProtocolT", bound=EnsembleGenerationProtocol)


@dataclass
class EnsembleFileSysOperationSet:
    """Dataclass to represent a set of Ensemble file system operation objects"""

    operations: t.List[EnsembleGenerationProtocol] = field(default_factory=list)
    """Set of Ensemble file system objects that match the EnsembleGenerationProtocol"""

    def add_copy(
        self, src: pathlib.Path, dest: t.Optional[pathlib.Path] = None
    ) -> None:
        """Add a copy operation to the operations list

        :param src: Path to source
        :param dest: Path to destination
        """
        self.operations.append(EnsembleCopyOperation(src, dest))

    def add_symlink(
        self, src: pathlib.Path, dest: t.Optional[pathlib.Path] = None
    ) -> None:
        """Add a symlink operation to the operations list

        :param src: Path to source
        :param dest: Path to destination
        """
        self.operations.append(EnsembleSymlinkOperation(src, dest))

    def add_configuration(
        self,
        src: pathlib.Path,
        file_parameters: t.Mapping[str,t.Sequence[str]],
        dest: t.Optional[pathlib.Path] = None,
        tag: t.Optional[str] = None,
    ) -> None:
        """Add a configure operation to the operations list

        :param src: Path to source
        :param file_parameters: File parameters to find and replace
        :param dest: Path to destination
        :param tag: Tag to use for find and replacement
        """
        self.operations.append(EnsembleConfigureOperation(src, file_parameters, dest, tag))

    @property
    def copy_operations(self) -> t.List[EnsembleCopyOperation]:
        """Property to get the list of copy files.

        :return: List of EnsembleCopyOperation objects
        """
        return self._filter(EnsembleCopyOperation)

    @property
    def symlink_operations(self) -> t.List[EnsembleSymlinkOperation]:
        """Property to get the list of symlink files.

        :return: List of EnsembleSymlinkOperation objects
        """
        return self._filter(EnsembleSymlinkOperation)

    @property
    def configure_operations(self) -> t.List[EnsembleConfigureOperation]:
        """Property to get the list of configure files.

        :return: List of EnsembleConfigureOperation objects
        """
        return self._filter(EnsembleConfigureOperation)

    def _filter(self, type: t.Type[EnsembleGenerationProtocolT]) -> t.List[EnsembleGenerationProtocolT]:
        """Filters the operations list to include only instances of the
        specified type.

        :param type: The type of operations to filter
        :return: A list of operations that are instances of the specified type
        """
        return [x for x in self.operations if isinstance(x, type)]