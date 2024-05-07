from smartsim import Experiment
from smartsim.log import get_logger

logger = get_logger("Experiment Log")
# Initialize the Experiment
exp = Experiment("getting-started", launcher="auto")

# Initialize a standalone Feature Store
standalone_feature_store = exp.create_feature_store(fs_nodes=1)

# Initialize a RunSettings object for Ensemble
ensemble_settings = exp.create_run_settings(exe="/path/to/executable_producer_simulation")

# Initialize Ensemble
producer_ensemble = exp.create_ensemble("producer", run_settings=ensemble_settings, replicas=2)

# Enable key prefixing for Ensemble members
producer_ensemble.enable_key_prefixing()

# Initialize a RunSettings object for Model
model_settings = exp.create_run_settings(exe="/path/to/executable_consumer_simulation")
# Initialize Model
consumer_model = exp.create_model("consumer", model_settings)

# Generate SmartSim entity folder tree
exp.generate(standalone_feature_store, producer_ensemble, consumer_model, overwrite=True)

# Launch Feature Store
exp.start(standalone_feature_store, summary=True)

# Launch Ensemble
exp.start(producer_ensemble, block=True, summary=True)

# Register Ensemble members on consumer Model
for model in producer_ensemble:
    consumer_model.register_incoming_entity(model)

# Launch consumer Model
exp.start(consumer_model, block=True, summary=True)

# Clobber Feature Store
exp.stop(standalone_feature_store)