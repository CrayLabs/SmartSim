from dataclasses import dataclass, field

@dataclass
class ParamSet():
    """
    Represents a set of file parameters and execution arguments.
    """
    _params: dict[str, str] = field(default_factory=dict)
    _exe_args: dict[str, list[str]] = field(default_factory=dict)