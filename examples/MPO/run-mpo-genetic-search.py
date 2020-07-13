from crayai import hpo
from smartsim import Experiment

exp = Experiment("MPO")
alloc = exp.get_allocation(nodes=20, partition="iv24", time="10:00:00")

# Define model parameter space
params = hpo.Params([["--KH", 2000, (0, 4000)],
                     ["--KHTH", 2000, (0, 4000)]])

# Define the evaluator
cmd = f"python -u eval-script.py --alloc {alloc}"

evaluator = hpo.Evaluator(cmd,
                          workload_manager='local',
                          num_parallel_evals=20,
                          verbose=True)


optimizer = hpo.genetic.Optimizer(evaluator,
                                  pop_size= 10,
                                  num_demes=2,
                                  generations=5,
                                  mutation_rate=0.05,
                                  crossover_rate=0.4,
                                  verbose=True )


# Run the optimizer over the model parameter space
optimizer.optimize(params)

exp.release()
