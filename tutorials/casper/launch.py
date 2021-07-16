#!/usr/bin/env python
import os, sys

import argparse, subprocess
from string import Template
from utils import run_cmd

def parse_command_line(args, description):
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--db-nodes", default=1,
                        help="Number of nodes for the SmartSim database, default=1")
    parser.add_argument("--ngpus-per-node", default=0,
                        help="Number of gpus per SmartSim database node, default=0")
    parser.add_argument("--walltime", default="00:30:00",
                        help="Total walltime for submitted job, default=00:30:00")
    parser.add_argument("--ensemble-size", default=1,
                        help="Number of ensemble members to run, default=1")
    parser.add_argument("--member-nodes", default=1,
                        help="Number of nodes per ensemble member, default=1")
    parser.add_argument("--account", default="P93300606",
                        help="Account ID")
    parser.add_argument("--db-port", default=6780,
                        help="db port, default=6780")

    args = parser.parse_args(args[1:])
    ngpus = ""
    if int(args.ngpus_per_node) > 0:
        ngpus = ":ngpus="+args.ngpus_per_node


    return {"db_nodes":args.db_nodes, "ngpus": ngpus, "client_nodes": args.ensemble_size*args.member_nodes, 
            "walltime": args.walltime, "account" : args.account, "member_nodes": args.member_nodes, 
            "ensemble_size": args.ensemble_size, "db_port": args.db_port, "python_sys_path": sys.path}

def _main_func(desc):
    templatevars = parse_command_line(sys.argv, desc)
    
    template_files = ["resv_job.template", "launch_database_cluster.template", "launch_client.template"]
    
    for template in template_files:
        with open(template) as f:
            src = Template(f.read())
            result = src.safe_substitute(templatevars)
        result_file = template.replace("template","sh")
        with open(result_file, "w") as f:
            f.write(result)

    run_cmd("qsub resv_job.sh", verbose=True)

if __name__ == "__main__":
    _main_func(__doc__)
