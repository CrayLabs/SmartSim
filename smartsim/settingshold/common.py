import typing as t

IntegerArgument = t.Dict[str, t.Optional[int]]
StringArgument = t.Dict[str, t.Optional[str]]

def process_env_vars(env_vars: StringArgument) -> None:
    """Validate that user passed env vars are of correct type.
    """
    for key, value in env_vars.items():
        if not isinstance(value, str) and value is not None:
            raise TypeError(f"Value for '{key}' must be a string.")

def process_args(launch_args: t.Dict[str, t.Union[str,int,float,None]]) -> None:
    """Validate that user passed launch args and scheduler args are of correct type.
    """
    for key, value in launch_args.items():
        if not isinstance(value, (str,int,float)) and value is not None:
            raise TypeError(f"Value for '{key}' must be a string.")
    