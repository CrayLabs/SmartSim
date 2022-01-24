from smartsim.slurm import _get_alloc_cmd


def test_get_alloc_format():
    time = "10:00:00"
    nodes = 5
    account = "A35311"
    options = {"ntasks-per-node": 5}
    alloc_cmd = _get_alloc_cmd(nodes, time, account, options)
    result = [
        "--no-shell",
        "-N",
        "5",
        "-J",
        "SmartSim",
        "-t",
        "10:00:00",
        "-A",
        "A35311",
        "--ntasks-per-node=5",
    ]
    assert alloc_cmd == result


def test_get_alloc_format_overlap():
    """Test get alloc with collision between arguments and options"""
    time = "10:00:00"
    nodes = 5
    account = "A35311"
    options = {"N": 10, "time": "15", "account": "S1242"}
    alloc_cmd = _get_alloc_cmd(nodes, time, account, options)
    result = [
        "--no-shell",
        "-N",
        "5",
        "-J",
        "SmartSim",
        "-t",
        "10:00:00",
        "-A",
        "A35311",
    ]
    assert result == alloc_cmd
