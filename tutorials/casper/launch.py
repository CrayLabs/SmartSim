#!/usr/bin/env python
import os, sys

import argparse, subprocess
from string import Template

def parse_command_line(args, description):
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--db-nodes", default=1,
                        help="Number of nodes for the SmartSim database")
    parser.add_argument("--ngpus-per-node", default=0,
                        help="Number of gpus per SmartSim database node")
    parser.add_argument("--walltime", default="00:30:00",
                        help="Total walltime for submitted job")
    parser.add_argument("--ensemble-size", default=1,
                        help="Number of ensemble members to run")
    parser.add_argument("--member-nodes", default=1,
                        help="Number of nodes per ensemble member")
    parser.add_argument("--account", default="P93300606",
                        help="Account ID")
    parser.add_argument("--db-port", default=6780,
                        help="db port")

    args = parser.parse_args(args[1:])
    ngpus = ""
    if int(args.ngpus_per_node) > 0:
        ngpus = ":ngpus="+args.ngpus_per_node


    return {"db_nodes":args.db_nodes, "ngpus": ngpus, "client_nodes": args.ensemble_size*args.member_nodes, 
            "walltime": args.walltime, "account" : args.account, "member_nodes": args.member_nodes, 
            "ensemble_size": args.ensemble_size, "db_port": args.db_port}

def execute(command):
    """
    Function for running a command on shell.
    Args:
        command (str):
            command that we want to run.
    Raises:
        Error with the return code from shell.
    """
    print ('\n',' >>  ',*command,'\n')

    try:
        subprocess.check_call(command, stdout=sys.stdout, stderr=subprocess.STDOUT)

    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))


def _main_func(desc):
    templatevars = parse_command_line(sys.argv, desc)
    
    template_files = ["resv_job.template", "launch_database_cluster.template", "launch_client.template", "cleanup.template"]
    
    for template in template_files:
        with open(template) as f:
            src = Template(f.read())
            result = src.safe_substitute(templatevars)
        result_file = template.replace("template","sh")
        with open(result_file, "w") as f:
            f.write(result)

    execute(['qsub', 'resv_job.sh'])

if __name__ == "__main__":
    _main_func(__doc__)
