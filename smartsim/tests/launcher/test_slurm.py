import pytest
from shutil import which
from smartsim.launcher.SlurmLauncher import SlurmLauncher
from smartsim.error.errors import LauncherError

# skip if not on a slurm system
if not which("srun"):
    pytestmark = pytest.mark.skip()

def test_get_system_partition_info():
    """ This test ensures that _get_system_partition_info
        is able to retrieve at least one partition with
        non_zero node count and processors per node count.
    """
    slurm = SlurmLauncher()
    partitions = slurm._get_system_partition_info()

    assert(len(partitions)>0)

    for p_name, p_obj in partitions.items():
        assert(p_obj._is_valid_partition())

def test_all_partitions_valid():
    """This test checks that all partitions are valid."""

    slurm = SlurmLauncher()
    sys_partitions = slurm._get_system_partition_info()

    assert(len(sys_partitions)>0)

    for p_name, partition in sys_partitions.items():
        assert(partition._is_valid_partition())

def test_get_default_partition():
    """Test that there is a valid default partition."""
    slurm = SlurmLauncher()
    sys_partitions = slurm._get_system_partition_info()

    default_p_name = slurm._get_default_partition()

    assert(default_p_name in sys_partitions)
    assert(sys_partitions[default_p_name]._is_valid_partition())

def test_get_invalid_partition():
    """ This tests checks that an error is raised
        if an invalid partition is requested.
    """
    slurm = SlurmLauncher()
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
    slurm = SlurmLauncher()
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
    slurm = SlurmLauncher()
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
    slurm = SlurmLauncher()
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


slurm = SlurmLauncher()

def test_get_alloc():
    """Test getting an allocation on the default partition"""
    slurm.get_alloc(nodes=1)
    assert(len(slurm.alloc_manager().values()) == 1)
    slurm.free_alloc(list(slurm.alloc_manager().keys())[0])


# Error handling cases

def test_bad_partition_get_alloc():
    """Test getting an allocation on a non-existant partition"""
    with pytest.raises(LauncherError):
        slurm.get_alloc(nodes=1, partition="not-a-partition")
