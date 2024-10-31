import pathlib
import typing as t


def check_src_and_dest_path(
    src: pathlib.Path, dest: t.Union[pathlib.Path, None]
) -> None:
    """Validate that the provided source and destination paths are
    of type pathlib.Path. Additionally, validate that destination is a
    relative Path and source is a absolute Path.

    :param src: The source path to check
    :param dest: The destination path to check
    :raises TypeError: If either src or dest is not of type pathlib.Path
    :raises ValueError: If source is not an absolute Path or if destination is not
        a relative Path
    """
    if not isinstance(src, pathlib.Path):
        raise TypeError(f"src must be of type pathlib.Path, not {type(src).__name__}")
    if dest is not None and not isinstance(dest, pathlib.Path):
        raise TypeError(
            f"dest must be of type pathlib.Path or None, not {type(dest).__name__}"
        )
    if dest is not None and dest.is_absolute():
        raise ValueError(f"dest must be a relative Path")
    if not src.is_absolute():
        raise ValueError(f"src must be an absolute Path")
