from smartsim import Experiment
import os

nodes = 2
ppn = 2

experiment = Experiment("lammps_melt_analysis")
alloc = experiment.get_allocation(nodes=nodes+4, ppn=ppn)

run_settings = {
    "nodes": nodes,
    "ppn" : ppn,
    "executable": "lmp",
    "exe_args": "-i in.melt",
    "alloc": alloc
}
node_settings = {
    "nodes": 1,
    "executable": "python smartsim_node.py",
    "exe_args": f"--ranks={nodes*ppn} --time=250",
    "alloc": alloc
}

m1 = experiment.create_model("lammps_melt", run_settings=run_settings)
m1.attach_generator_files(to_copy=[f"{os.getcwd()}/in.melt"])
n1 = experiment.create_node("lammps_data_processor",run_settings=node_settings)
n1.attach_generator_files(to_copy=[f"{os.getcwd()}/smartsim_node.py"])
orc = experiment.create_orchestrator_cluster(alloc, overwrite=True)
experiment.generate()
experiment.start(orchestrator=orc)
experiment.poll()
experiment.stop()
experiment.release()
