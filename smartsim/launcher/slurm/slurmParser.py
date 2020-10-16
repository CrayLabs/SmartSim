def parse_salloc(output):
    for line in output.split("\n"):
        if line.startswith("salloc: Granted job allocation"):
            return line.split()[-1]


def parse_salloc_error(output):
    """Parse and return error output of a failed salloc command

    :param output: stderr output of salloc command
    :type output: str
    :return: error message
    :rtype: str
    """
    # look for error first
    for line in output.split("\n"):
        if line.startswith("salloc: error:"):
            error = line.split("error:")[1]
            return error.strip()
    # if no error line, take first line
    for line in output.split("\n"):
        if line.startswith("salloc: "):
            error = " ".join((line.split()[1:]))
            return error.strip()
    # if neither, present a base error message
    base_err = "Slurm allocation error"
    return base_err


def parse_sacct_step(output):
    """Parse the number of job steps launched on an allocation

    :param output: output of the sacct command
    :type output: str
    :return: number of job steps
    :rtype: int
    """
    line = output.strip().split("\n")[-1]
    job_id = line.split("|")[0]
    if "." not in job_id:
        return 0
    else:
        job, step = job_id.split(".")
        if step.startswith("ext"):
            return 0
        else:
            return int(step) + 1


def parse_sacct(output, job_id):
    """Parse and return output of the sacct command

    :param output: output of the sacct command
    :type output: str
    :param job_id: allocation id or job step id
    :type job_id: str
    :return: status and returncode
    :rtype: tuple
    """
    result = ("NOTFOUND", "NAN")
    for line in output.split("\n"):
        if line.strip().startswith(job_id):
            line = line.split("|")
            stat = line[1]
            code = line[2].split(":")[0]
            result = (stat, code)
            break
    return result


def parse_sstat_nodes(output):
    """Parse and return the sstat command

    This function parses and returns the nodes of
    a job in a list with the duplicates removed.

    :param output: output of the sstat command
    :type output: str
    :return: compute nodes of the allocation or job
    :rtype: list of str
    """
    nodes = []
    for line in output.split("\n"):
        sstat_string = line.split("|")

        # sometimes there are \n that we need to ignore
        if len(sstat_string) >= 2:
            node = sstat_string[1]
            nodes.append(node)
    return list(set(nodes))


def parse_step_id_from_sacct(output, step_name):
    """Parse and return the step id from a sacct command

    :param output: output of sacct --noheader -p
                   --format=jobname,jobid --job <alloc>
    :type output: str
    :param step_name: the name of the step to query
    :type step_name: str
    :return: the step_id
    :rtype: str
    """
    step_id = None
    for line in output.split("\n"):
        sacct_string = line.split("|")
        if len(sacct_string) < 2:
            continue
        if sacct_string[0] == step_name:
            step_id = sacct_string[1]
    return step_id
