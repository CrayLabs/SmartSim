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
        Due to the reliance on executing an sinfo command,
        this test could fail because of system configuration
        as well as code implementation.
    """
    sl = SlurmLauncher()
    partitions = sl._get_system_partition_info()

    assert(len(partitions)>0)

    for p_name, partition in partitions.items():
        assert(partition._is_valid_partition())

def test_validate_one_partition():
    """This test checks that the validate() function
       affirms that a proper node and ppn request
       for a partition does not fail.
    """
    sl = SlurmLauncher()
    sys_partitions = sl._get_system_partition_info()

    assert(len(sys_partitions)>0)

    p_name = list(sys_partitions.keys())[0]
    n_nodes = len(sys_partitions[p_name].nodes)
    ppn = sys_partitions[p_name].min_ppn
    print(p_name, n_nodes, ppn)

    assert(sys_partitions[p_name]._is_valid_partition())
    sl.validate(nodes=n_nodes-1, ppn=ppn-1, partition=p_name)

def test_validate_fail_nodes_one_partition():
    """This test checks that the validate() function
       raises an error when too many nodes are requested
       for a single partition.
    """
    sl = SlurmLauncher()
    sys_partitions = sl._get_system_partition_info()

    assert(len(sys_partitions)>0)

    p_name = list(sys_partitions.keys())[0]
    n_nodes = len(sys_partitions[p_name].nodes)
    ppn = sys_partitions[p_name].min_ppn

    assert(sys_partitions[p_name]._is_valid_partition())
    with pytest.raises(LauncherError):
        sl.validate(nodes=n_nodes+1, ppn=ppn, partition=p_name)

def test_validate_fail_ppn_one_partition():
    """This test checks that the validate() function
       raises an error when too many ppn are requested
       for a single partition.
    """
    sl = SlurmLauncher()
    sys_partitions = sl._get_system_partition_info()

    assert(len(sys_partitions)>0)

    p_name = list(sys_partitions.keys())[0]
    n_nodes = len(sys_partitions[p_name].nodes)
    ppn = sys_partitions[p_name].min_ppn

    assert(sys_partitions[p_name]._is_valid_partition())
    with pytest.raises(LauncherError):
        sl.validate(nodes=n_nodes, ppn=ppn+1, partition=p_name)

def test_validate_fail_nodes_all_partitions():
    """This test checks that the validate() function
       raises an error when too many nodes are requested
       over all partitions.
       :raises: LauncherError
    """

    sl = SlurmLauncher()
    sys_partitions = sl._get_system_partition_info()

    if(len(sys_partitions)<=1):
        pytest.skip()

    nodes = set()
    p_names = []
    for p_name, partition in sys_partitions.items():
        assert(partition._is_valid_partition())
        nodes = nodes.union(partition.nodes)
        p_names.append(p_name)
    n_nodes = len(nodes)

    with pytest.raises(LauncherError):
        [sl.validate(nodes=n_nodes+1, ppn=1, partition=p_name) for p_name in p_names]

def test_validate_fail_ppn_all_partitions():
    """This test checks that the validate() function
       raises an error when too many ppn are requested
       over all partitions.
       :raises: LauncherError
    """

    sl = SlurmLauncher()
    sys_partitions = sl._get_system_partition_info()

    if(len(sys_partitions)<=1):
        pytest.skip()

    nodes=set()
    p_names = []
    min_ppn = None

    for p_name, partition in sys_partitions.items():
        assert(partition._is_valid_partition())
        nodes = nodes.union(partition.nodes)
        if min_ppn is None:
            min_ppn = partition.min_ppn
        else:
            min_ppn = min(min_ppn, partition.min_ppn)
        p_names.append(p_name)
    n_nodes=len(nodes)

    with pytest.raises(LauncherError):
        [sl.validate(nodes=n_nodes, ppn=min_ppn+1, partition=p_name) for p_name in p_names]


def test_run_on_alloc():
    pass

def test_get_alloc():
    pass

def test_get_job_status():
    pass

def test_free_alloc():
    pass