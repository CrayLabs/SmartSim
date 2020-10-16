import os
import pytest
from shutil import which
from smartsim.launcher import SlurmLauncher
from smartsim.error.errors import LauncherError

# skip if not on a slurm system
if not which("srun"):
    pytestmark = pytest.mark.skip()

slurm = SlurmLauncher()

def test_get_system_partition_info():
    """ This test ensures that _get_system_partition_info
        is able to retrieve at least one partition with
        non_zero node count and processors per node count.
    """
    partitions = slurm._get_system_partition_info()
    assert(len(partitions)>0)
    for p_name, p_obj in partitions.items():
        assert(p_obj._is_valid_partition())

def test_get_default_partition():
    """Test that there is a valid default partition."""
    sys_partitions = slurm._get_system_partition_info()
    default_p_name = slurm._get_default_partition()
    assert(default_p_name in sys_partitions)
    assert(sys_partitions[default_p_name]._is_valid_partition())

def test_get_invalid_partition():
    """ This tests checks that an error is raised
        if an invalid partition is requested.
    """
    sys_partitions = slurm._get_system_partition_info()
    assert(len(sys_partitions)>0)

    # Create a partition name that is not valid
    valid_p_names = set()
    for p_name, p_obj in sys_partitions.items():
        valid_p_names.add(p_obj.name)

    invalid_p_name = ""
    for valid_p_name in valid_p_names:
        invalid_p_name += valid_p_name
        if not invalid_p_name in valid_p_names:
            break

    with pytest.raises(LauncherError):
        slurm.validate(nodes=1, ppn=1, partition=invalid_p_name)

def test_validate_one_partition():
    """This test checks that the validate() function
       affirms that a proper node and ppn request
       for a partition does not fail.
    """
    sys_partitions = slurm._get_system_partition_info()
    assert(len(sys_partitions)>0)

    p_name = list(sys_partitions.keys())[0]
    n_nodes = len(sys_partitions[p_name].nodes)
    partition = sys_partitions[p_name]
    assert(partition._is_valid_partition())

    n_ppn_nodes = 0
    desired_ppn = 1
    for node in partition.nodes:
        if(node.ppn >= desired_ppn):
            n_ppn_nodes += 1

    slurm.validate(nodes=n_ppn_nodes, ppn=desired_ppn, partition=p_name)

def test_validate_fail_nodes_one_partition():
    """This test checks that the validate() function
       raises an error when too many nodes are requested
       for a given ppn.
    """
    sys_partitions = slurm._get_system_partition_info()
    assert(len(sys_partitions)>0)

    p_name = list(sys_partitions.keys())[0]
    n_nodes = len(sys_partitions[p_name].nodes)
    partition = sys_partitions[p_name]
    assert(partition._is_valid_partition())

    n_ppn_nodes = 0
    desired_ppn = 16
    for node in sys_partitions[p_name].nodes:
        if(node.ppn >= desired_ppn):
            n_ppn_nodes += 1

    with pytest.raises(LauncherError):
        slurm.validate(nodes=n_ppn_nodes+1, ppn=desired_ppn, partition=p_name)

def test_validate_fail_ppn_one_partition():
    """This test checks that the validate() function
       raises an error when zero nodes on the requested
       partition have sufficient ppn.
    """
    sys_partitions = slurm._get_system_partition_info()
    assert(len(sys_partitions)>0)

    p_name = list(sys_partitions.keys())[0]
    n_nodes = len(sys_partitions[p_name].nodes)
    n_ppn_nodes = 0
    desired_ppn = 16
    assert(sys_partitions[p_name]._is_valid_partition())

    max_ppn = 0
    for node in sys_partitions[p_name].nodes:
        if(node.ppn >= max_ppn):
            max_ppn = node.ppn

    with pytest.raises(LauncherError):
        slurm.validate(nodes=1, ppn=max_ppn+1, partition=p_name)


# ------------------------------------------------------------------
# Error handling cases

def test_get_step_status():
    """test calling get_step_status for step that doesnt exist"""
    status = slurm.get_step_status(11111.1)
    assert(status.status == "NOTFOUND")
    assert(status.returncode == "NAN")
    assert(status.error == None)
    assert(status.output == None)


def test_bad_get_step_nodes():
    """test call of get_step_nodes with a step that doesnt exist"""
    with pytest.raises(LauncherError):
        slurm.get_step_nodes(111111.1)

def test_no_alloc_create_step():
    """create_step called without an alloc in run_settings"""
    cwd = os.path.dirname(os.path.abspath(__file__))
    run_settings = {
        "nodes": 1,
        "ppn": 1,
        "out_file": cwd + "/out.txt",
        "err_file": cwd + "/err.txt",
        "cwd": cwd,
        "executable": "a.out",
        "exe_args": "--input"
    }
    with pytest.raises(LauncherError):
        slurm.create_step("test", run_settings)

@pytest.mark.skip(reason="We currently don't test that a user's inputted allocation exists")
def test_bad_alloc_create_step():
    """create_step called with a non-existant allocation"""
    cwd = os.path.dirname(os.path.abspath(__file__))
    run_settings = {
        "nodes": 1,
        "ppn": 1,
        "out_file": cwd + "/out.txt",
        "err_file": cwd + "/err.txt",
        "cwd": cwd,
        "executable": "a.out",
        "exe_args": "--input",
        "alloc": 1111111
    }
    with pytest.raises(LauncherError):
        slurm.create_step("test", run_settings)
