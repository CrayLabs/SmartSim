from ...error import LauncherError
from ...utils import get_env, get_logger
from .slurmCommands import salloc, scancel
from .slurmLauncher import SlurmLauncher
from .slurmParser import parse_salloc, parse_salloc_error

logger = get_logger(__name__)


def get_slurm_allocation(nodes=1, add_opts={}):
    """Request an allocation

    This function requests an allocation with the specified arguments.
    Anything passed to the add_opts will be processed as a Slurm
    argument and appended to the salloc command with the appropriate
    prefix (e.g. "-" or "--").

    The add_opts can be used to pass extra settings to the
    workload manager such as the following for Slurm:
        - nodelist="nid00004"

    For arguments without a value, pass None or and empty
    string as the value for the kwarg. For Slurm:
        - exclusive=None

    :param nodes: number of nodes for the allocation, defaults to 1
    :type nodes: int, optional
    :param add_opts: additional options for the slurm wlm
    :type add_opts: dict
    :raises LauncherError: if the allocation is not successful
    :return: the id of the allocation
    :rtype: str
    """
    SlurmLauncher.check_for_slurm()

    salloc_args = _get_alloc_cmd(nodes, add_opts=add_opts)
    debug_msg = " ".join(salloc_args[1:])
    logger.debug(f"Allocation settings: {debug_msg}")

    # TODO figure out why this goes to stderr
    _, err = salloc(salloc_args)
    alloc_id = parse_salloc(err)
    if alloc_id:
        logger.info("Allocation successful with Job ID: %s" % alloc_id)
    else:
        logger.debug(err)
        error = parse_salloc_error(err)
        if not error:
            logger.error(err)
            raise LauncherError("Slurm allocation error")
        else:
            raise LauncherError(error)
    return str(alloc_id)


def release_slurm_allocation(alloc_id):
    """Free an allocation from within the launcher

    :param alloc_id: allocation id
    :type alloc_id: str
    :raises LauncherError: if allocation not found within the AllocManager
    :raises LauncherError: if allocation could not be freed
    """
    SlurmLauncher.check_for_slurm()

    logger.info(f"Releasing allocation: {alloc_id}")
    returncode, _, err = scancel([str(alloc_id)])

    if returncode != 0:
        logger.error("Unable to revoke your allocation for jobid %s" % alloc_id)
        logger.error(
            "The job may have already timed out, or you may need to cancel the job manually"
        )
        raise LauncherError("Unable to revoke your allocation for jobid %s" % alloc_id)

    logger.info(f"Successfully freed allocation {alloc_id}")


def _get_alloc_cmd(nodes, add_opts={}):
    """Return the command to request an allocation from Slurm with
    the class variables as the slurm options."""

    salloc_args = [
        "--no-shell",
        "-N",
        str(nodes),
        "-J",
        "SmartSim",
    ]

    for opt, val in add_opts.items():
        short_arg = True if len(str(opt)) == 1 else False
        prefix = "-" if short_arg else "--"
        if not val:
            salloc_args += [prefix + opt]
        else:
            if short_arg:
                salloc_args += [prefix+opt, str(val)]
            else:
                salloc_args += ["=".join((prefix + opt, str(val)))]

    return salloc_args