from ..commands import Command
import typing as t
import sys
import pathlib
from dataclasses import dataclass, field

file_op_entry_point = "smartsim._core.entrypoints.file_operations"

class GenerationContext():
    """Context for file system generation operations."""
    def __init__(self, job_root_path: pathlib.Path):
        self.job_root_path = job_root_path
        """The Job root path"""


class GenerationProtocol(t.Protocol):
    """Protocol for Generation Operations."""
    def format(self) -> Command:
        """Return a formatted Command that can be executed by a Launcher"""

def create_final_dest(job_root_path: pathlib.Path, dest: pathlib.Path) -> str:
    return str(job_root_path / dest)

class CopyOperation(GenerationProtocol):
    """Copy Operation"""
    def __init__(self, src: pathlib.Path, dest: t.Union[pathlib.Path, None]) -> None:
        self.src = src
        self.dest = dest

    def format(self, context: GenerationContext) -> Command:
        """Create Command to invoke copy fs entry point"""
        final_dest = create_final_dest(context.job_root_path, self.dest)
        return Command([sys.executable, "-m", file_op_entry_point,
                        "copy", self.src, final_dest])


class SymlinkOperation(GenerationProtocol):
    """Symlink Operation"""
    def __init__(self, src: pathlib.Path, dest: t.Union[pathlib.Path, None]) -> None:
        self.src = src
        self.dest = dest

    def format(self, context: GenerationContext) -> Command:
        """Create Command to invoke symlink fs entry point"""
        final_dest = create_final_dest(context.job_root_path, self.dest)
        return Command([sys.executable, "-m", file_op_entry_point,
                        "symlink", self.src, final_dest])


class ConfigureOperation(GenerationProtocol):
    """Configure Operation"""
    def __init__(self, src: pathlib.Path, dest: t.Union[pathlib.Path, None], tag: t.Optional[str] = None) -> None:
        self.src = src
        self.dest = dest
        self.tag = tag if tag else ";"
    
    # TODO discuss format as function name
    def format(self, context: GenerationContext) -> Command:
        """Create Command to invoke configure fs entry point"""
        final_dest = create_final_dest(context.job_root_path, self.dest)
        return Command([sys.executable, "-m", file_op_entry_point,
                        "configure", self.src, final_dest, self.tag, "encoded_dict"])

@dataclass
class FileSysOperationSet():
    """Dataclass to represent a set of FS Operation Objects"""
    
    # disallow modification - dunder function (post ticket to reevaluate API objects)
    operations: t.List[GenerationContext] = field(default_factory=list)
    """Set of FS Objects that match the GenerationProtocol"""
    
    def add_copy(self, src: pathlib.Path, dest: t.Optional[pathlib.Path] = None) -> None:
        """Add a copy operation to the operations list"""
        self.operations.append(CopyOperation(src, dest))

    def add_symlink(self, src: pathlib.Path, dest: t.Optional[pathlib.Path] = None) -> None:
        """Add a symlink operation to the operations list"""
        self.operations.append(SymlinkOperation(src, dest))

    def add_configuration(self, src: pathlib.Path, dest: t.Optional[pathlib.Path] = None, tag: t.Optional[str] = None) -> None:
        """Add a configure operation to the operations list"""
        self.operations.append(ConfigureOperation(src, dest, tag))
    
    # entirely for introspection
    # create generic filter operation that takes in a class type -> filter() -> public
    # properties will call filter function - t.List of operation objects
    @property
    def copy_operations(self) -> t.List[CopyOperation]:
        """Property to get the list of copy files.""" # return dict instead of operation list
        return [x for x in self.operations if isinstance(x, CopyOperation)]
    
    @property
    def symlink_operations(self) -> t.List[SymlinkOperation]:
        """Property to get the list of symlink files."""
        return [x for x in self.operations if isinstance(x, SymlinkOperation)]
    
    @property
    def configure_operations(self) -> t.List[ConfigureOperation]:
        """Property to get the list of configure files."""
        return [x for x in self.operations if isinstance(x, ConfigureOperation)]