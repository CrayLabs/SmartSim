from smartsim.launcher.lsf import lsfParser

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


def test_parse_bjobs_jobid():
    """Parse jobid from bjobs called with -w"""
    output = (
        "JOBID   USER       STAT   SLOTS    QUEUE       START_TIME    FINISH_TIME   JOB_NAME                      \n"
        "1234567 smartsim   RUN    85       debug       -             -             SmartSim   \n"
    )
    parsed_id = lsfParser.parse_step_id_from_bjobs(output, step_name="SmartSim")
    assert parsed_id == "1234567"
