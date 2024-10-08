import pathlib
import typing as t
from dataclasses import dataclass, field



class EnsembleGenerationProtocol(t.Protocol):
    """Protocol for Generation Operations Ensemble."""
    src: pathlib.Path
    dest: t.Optional[pathlib.Path]


class EnsembleCopyOperation(EnsembleGenerationProtocol):
    """Copy Operation"""

    def __init__(
        self, src: pathlib.Path, dest: t.Optional[pathlib.Path] = None
    ) -> None:
        self.src = src
        self.dest = dest


class EnsembleSymlinkOperation(EnsembleGenerationProtocol):
    """Symlink Operation"""

    def __init__(self, src: pathlib.Path, dest: t.Optional[pathlib.Path] = None) -> None:
        self.src = src
        self.dest = dest


class EnsembleConfigureOperation(EnsembleGenerationProtocol):
    """Configure Operation"""

    def __init__(
        self,
        src: pathlib.Path,
        file_parameters:t.Mapping[str,t.Sequence[str]],
        dest: t.Optional[pathlib.Path] = None,
        tag: t.Optional[str] = None,
    ) -> None:
        self.src = src
        self.dest = dest
        self.file_parameters = file_parameters
        self.tag = tag if tag else ";"


U = t.TypeVar("U", bound=EnsembleGenerationProtocol)


@dataclass
class EnsembleFileSysOperationSet:
    """Dataclass to represent a set of FS Operation Objects"""

    operations: t.List[EnsembleGenerationProtocol] = field(default_factory=list)
    """Set of FS Objects that match the GenerationProtocol"""

    def add_copy(
        self, src: pathlib.Path, dest: t.Optional[pathlib.Path] = None
    ) -> None:
        """Add a copy operation to the operations list"""
        self.operations.append(EnsembleCopyOperation(src, dest))

    def add_symlink(
        self, src: pathlib.Path, dest: t.Optional[pathlib.Path] = None
    ) -> None:
        """Add a symlink operation to the operations list"""
        self.operations.append(EnsembleSymlinkOperation(src, dest))

    def add_configuration(
        self,
        src: pathlib.Path,
        file_parameters: t.Mapping[str,t.Sequence[str]],
        dest: t.Optional[pathlib.Path] = None,
        tag: t.Optional[str] = None,
    ) -> None:
        """Add a configure operation to the operations list"""
        self.operations.append(EnsembleConfigureOperation(src, file_parameters, dest, tag))

    @property
    def copy_operations(self) -> t.List[EnsembleCopyOperation]:
        """Property to get the list of copy files."""
        return self._filter(EnsembleCopyOperation)

    @property
    def symlink_operations(self) -> t.List[EnsembleSymlinkOperation]:
        """Property to get the list of symlink files."""
        return self._filter(EnsembleSymlinkOperation)

    @property
    def configure_operations(self) -> t.List[EnsembleConfigureOperation]:
        """Property to get the list of configure files."""
        return self._filter(EnsembleConfigureOperation)

    def _filter(self, type: t.Type[U]) -> t.List[U]:
        return [x for x in self.operations if isinstance(x, type)]