
# Training a surrogate model

In this example, a neural network is trained to act like a surrogate model and to solve a
well-known physical problem, i.e. computing the steady state of heat diffusion. The training
dataset is constructed by running simulations *while* the model is being trained.

The notebook also displays how the surrogate model prediction improves during training.
