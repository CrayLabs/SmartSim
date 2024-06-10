from dataclasses import dataclass

@dataclass
class ParamSet():
    _params: dict[str, str]
    _exe_args: dict[str, list[str]]