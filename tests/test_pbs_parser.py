from os.path import dirname
from pathlib import Path

from smartsim._core.launcher.pbs import pbsParser

# -- qsub ---------------------------------------------------------


def test_parse_qsub():
    output = "12345.sdb"
    step_id = pbsParser.parse_qsub(output)
    assert step_id == "12345.sdb"


def test_parse_qsub_error():
    output = "qsub: Unknown queue"
    error = "Unknown queue"
    parsed_error = pbsParser.parse_qsub_error(output)
    assert error == parsed_error


# -- qstat ---------------------------------------------------------


def test_parse_qstat_nodes(fileutils):
    """Parse nodes from qsub called with -f -F json"""
    file_path = fileutils.get_test_conf_path("qstat.json")
    output = Path(file_path).read_text()
    nodes = ["server_1", "server_2"]
    parsed_nodes = pbsParser.parse_qstat_nodes(output)
    assert nodes == parsed_nodes


def test_parse_qstat_status():
    """test retrieval of status and exitcode"""
    output = (
        "Job id            Name             User              Time Use S Queue\n"
        "----------------  ---------------- ----------------  -------- - -----\n"
        "1289903.sdb       jobname          username          00:00:00 R queue\n"
    )
    status = "R"
    parsed_status = pbsParser.parse_qstat_jobid(output, "1289903.sdb")
    assert status == parsed_status
