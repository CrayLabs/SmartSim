from smartsim._core.launcher.lsf import lsfParser

# -- bsub ---------------------------------------------------------


def test_parse_bsub():
    output = "Job <12345> is submitted to queue <debug>."
    step_id = lsfParser.parse_bsub(output)
    assert step_id == "12345"


def test_parse_bsub_error():
    output = (
        "Batch job not submitted. Please address the following and resubmit: \n"
        "--------------------------------------------------------------------\n\n"
        "**  Project not specified.  Please specify a project using the -P flag. \n\n"
        "**  Not a member of the specified project: .  You are currently a member of the following projects:\n"
        "       ABC123 \n\n"
        "------------------------------------------------------------------\n"
        "Please contact help@olcf.ornl.gov if you need assistance.\n\n"
        "Request aborted by esub. Job not submitted.\n"
    )
    error = (
        "Project not specified.  Please specify a project using the -P flag.\n"
        "Not a member of the specified project: .  You are currently a member of the following projects:\n"
        "ABC123"
    )
    parsed_error = lsfParser.parse_bsub_error(output)
    assert error == parsed_error

    output = "NOT A PARSABLE ERROR\nBUT STILL AN ERROR MESSAGE"
    parsed_error = lsfParser.parse_bsub_error(output)
    assert output == parsed_error

    output = "     \n"
    parsed_error = lsfParser.parse_bsub_error(output)
    assert parsed_error == "LSF run error"


# -- bjobs ---------------------------------------------------------


def test_parse_bsub_nodes(fileutils):
    """Parse nodes from bjobs called with -w"""
    output = (
        "JOBID   USER    STAT  QUEUE	 FROM_HOST   EXEC_HOST   JOB_NAME   SUBMIT_TIME\n"
        "1234567 smartsim RUN   batch	  login1      batch3:a01n02:a01n02:a01n02:a01n02:a01n02:a01n06:a01n06:a01n06:a01n06:a01n06 SmartSim Jul 24 12:53\n"
    )
    nodes = ["batch3", "a01n02", "a01n06"]
    parsed_nodes = lsfParser.parse_bjobs_nodes(output)
    assert nodes == parsed_nodes


def test_parse_max_step_id():
    """Get max step id from jslist"""
    output = (
        "         parent                cpus      gpus      exit               \n"
        "ID   ID       nrs    per RS    per RS    status         status\n"
        "===============================================================================\n"
        "    3    0         1         1         0         1       Complete\n"
        "    6    0         1         1         0         0       Complete\n"
        "    7    0         1         1         0         0       Complete\n"
        "    8    0         1         1         0         0       Complete\n"
        "    9    0         1         1         0         0       Complete\n"
        "    2    0         3   various   various       137         Killed\n"
        "    4    0         1   various   various       137         Killed\n"
        "    5    0         3   various   various       137         Killed\n"
    )
    parsed_id = lsfParser.parse_max_step_id_from_jslist(output)
    assert parsed_id == "9"


def test_parse_jslist():
    """Parse status and return code from jslist."""
    output = (
        "****************************************************************************\n"
        "   parent                cpus      gpus      exit  \n"
        "ID   ID       nrs    per RS    per RS    status         status\n"
        "===============================================================================\n"
        "    2    1       168         1         1         1       Complete\n"
        "    3    1         1         1         1         1       Complete\n"
        "    4    1       168         1         1         1       Complete\n"
        "    5    1         1         1         1         1       Complete\n"
        "    6    1       168         1         1         1       Complete\n"
        "    7    1         1         1         1         1       Complete\n"
        "    8    1       168         1         1         1       Complete\n"
        "    9    1         1         1         1         1       Complete\n"
        "   10    1       168         1         1         1       Complete\n"
        "    1    1         4   various   various         0        Running\n"
        "   11    1         1         1         1         1        Running\n"
    )
    parsed_result = lsfParser.parse_jslist_stepid(output, "1")
    result = ("Running", "0")
    assert parsed_result == result
