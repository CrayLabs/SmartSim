from crayai import hpo
from smartsim import Experiment

exp = Experiment("MPO")
alloc = exp.get_allocation(nodes=20, partition="iv24",
                           time="10:00:00", exclusive=None)

# Define model parameters and ranges
params = hpo.Params([["--KH", 2000, (0, 4000)],
                     ["--KHTH", 2000, (0, 4000)]])

# Define the evaluator
cmd = f"python -u eval-script.py --alloc {alloc}"

evaluator = hpo.Evaluator(cmd,
                          workload_manager='local',
                          num_parallel_evals=20,
                          verbose=True)

optimizer = hpo.GridOptimizer(evaluator,
                              verbose=True,
                              grid_size=10,
                              chunk_size=20)

# Run the optimizer over the model parameters
optimizer.optimize(params)

exp.release()
