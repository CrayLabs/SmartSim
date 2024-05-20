import typing as t

IntegerArgument = t.Dict[str, t.Optional[int]]
FloatArgument = t.Dict[str, t.Optional[float]]
StringArgument = t.Dict[str, t.Optional[str]]

def process_env_vars(env_vars: StringArgument) -> None:
    for key, value in env_vars.items():
        if not isinstance(value, str):
            raise ValueError(f"Value for '{key}' must be a string.")

# def process_launcher_args(launcher_args: t.Dict[str, t.Union[str,int,float,None]]):
    