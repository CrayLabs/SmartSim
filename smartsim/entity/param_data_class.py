from dataclasses import dataclass, field

@dataclass
class ParamSet():
    _params: dict[str, str] = field(default_factory=dict)
    _exe_args: dict[str, list[str]] = field(default_factory=dict)