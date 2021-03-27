
def parse_cobalt_step_status(output, step_id):
    status = "NOTFOUND"
    for line in output.split("\n"):
        if line.strip().startswith(step_id):
            line = line.split()
            status = line[1]
            break
    return status

def parse_cobalt_step_id(output, step_name):
    """Parse and return the step id from a cobalt qstat command

    :param output: output qstat
    :type output: str
    :param step_name: the name of the step to query
    :type step_name: str
    :return: the step_id
    :rtype: str
    """
    step_id = None
    for line in output.split("\n"):
        if line.strip().startswith(step_name):
            line = line.split()
            step_id = line[1]
            break
    return step_id

def parse_qsub_out(output):
    step_id = None
    for line in output.split("\n"):
        try:
            step_id = int(line.strip())
            break
        except ValueError:
            continue
    return str(step_id)
