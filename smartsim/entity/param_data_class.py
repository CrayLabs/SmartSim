from dataclasses import dataclass

@dataclass
class ParamSet():
    self._params: dict[str, str]
    self._exe_args: dict[str, list[str]]

    def add_exe_arg(name: str, value: str):
        self._exe_args[name] = value
    
    def add_params(name: str, value: str):
        self._params[name] = value