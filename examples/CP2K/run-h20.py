from smartsim import Experiment

# init experiment
experiment = Experiment("h2o")

# create ensembles
ensemble_params = {"STEPS": [10, 15, 20, 25]}
run_settings = {"executable": "cp2k.psmp",
                "partition": "gpu",
                "exe_args": "-i h2o.inp",
                "nodes": 1}
experiment.create_ensemble("h2o-1", params=ensemble_params, run_settings=run_settings)
experiment.generate(model_files="./h2o.inp")
experiment.start(launcher="slurm")
experiment.poll()
experiment.release()


