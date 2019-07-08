# Import the hpo submodule
from crayai import hpo


# Define hyperparameter and ranges
params = hpo.Params([["--learning_rate", 0.01, (1e-5, .5)],
                     ["--layer_1", 256, (64, 512)],
                     ["--layer_2", 64, (10, 128)],
                     ["--epoch", 100, (20, 400)],
                     ["--batch", 100, (5, 400)]])

# Define the evaluator
cmd = "python3 Deep-Regression-NN.py"
evaluator = hpo.Evaluator(cmd,
                          nodes=20,
                          launcher='urika',
                          urika_args="--no-node-list",
                          verbose=True)
"""
optimizer = hpo.genetic.Optimizer(evaluator, # Evaluator instance
                                 generations,        # Opt: Number of generations.
                                 num_demes,          # Opt: Number of distinct demes (populations)
                                 pop_size,           # Opt: Number of individuals per deme
                                 mutation_rate,      # Opt: Probability of mutation per
                                                     #      hyperparameter during creation of next
                                                     #      generation
                                 crossover_rate,     # Opt: Probability of crossover per
                                                     #      hyperparameter during creation of next
                                                     #      generation
                                 migration_interval, # Opt: Interval of migration between demes
                                 log_fn,             # Opt: Filename to record results of optimization
                                 verbose)            # Opt: Enable verbose output

optimizer = hpo.random.Optimizer(evaluator, # Evaluator instance
                              numIters,  # Opt: Number of iterations to run
                              seed,      # Opt: Seed for random number generator. Defaults to 0,
                                         #      i.e. random seed used.
                              verbose)   # Opt: Enable verbose output

optimizer = hpo.grid.Optimizer(evaluator,  # Evaluator instance
                                grid_size,  # Opt: Number of grid points to discretize for each
                                            #      hyperparameter
                                chunk_size, # Opt: Number of grid points to evaluate per batch (chunk)
                                verbose)    # Opt: Enable verbose output

"""

# Define genetic optimizer
optimizer = hpo.genetic.Optimizer(evaluator,
                                  pop_size=20,
                                  num_demes=4,
                                  generations=25,
                                  mutation_rate=0.30,
                                  crossover_rate=0.4,
                                  verbose=True)

# define random Optimizer
#optimizer = hpo.random.Optimizer(evaluator,
#                                 num_iters=100,
#                                 verbose=True)

# Run the optimizer over the provided hyperparameters
optimizer.optimize(params)
