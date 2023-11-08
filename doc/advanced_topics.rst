###############
Advanced Topics
###############

In this section, we illustrate some topics which experienced
users are expected to use in HPC workloads, especially when
simulation and AI are required to interact. The topics are
explained through code snippets,
with code which goes beyond SmartSim and SmartRedis API
(e.g. code showing how to jit-script a PyTorch model): the
intention is that of showing *one* simple way of leveraging
a feature, and they can be potentially optimized in
several ways. Examples are written in Python, but the same
result can be achieved with any SmartRedis client (C, C++,
Fortran and Python). Please refer to SmartRedis API
for language-specific details.

Using ML models on the DB
=========================

The combination of SmartSim and SmartRedis allows users
to store more than simple tensors on the DB. In the following
subsections, we show how to upload executable code, in the
form of ML models or functions, to the DB.
The stored code can then be run on stored tensors, and
the output is stored on the DB as well, where it can be
downloaded with standard ``get_tensor`` calls.

In general, there are two ways to upload serialized code
to the DB: from memory and from file. In all examples, we
will assume that a SmartSim ``Orchestator`` is up and running,
and that the code we will show is run as part of a SmartSim-launched
application ``Model``.


TensorFlow
----------
SmartSim provides :ref:`two helper methods for serializing
TensorFlow and Keras models <smartsim_tf_api>`: ``freeze_model`` and
``serialize_model``.

The method ``freeze_model`` is thought to be used in conjunction
with SmartRedis ``set_model_from_file``. The following is a typical
workflow, first we define the model:

.. code-block:: python

    import numpy as np
    from smartredis import Client
    from tensorflow import keras
    from smartsim.ml.tf import freeze_model

    model = keras.Sequential(
        layers=[
            keras.layers.InputLayer(input_shape=(28, 28), name="input"),
            keras.layers.Flatten(input_shape=(28, 28), name="flatten"),
            keras.layers.Dense(128, activation="relu", name="dense"),
            keras.layers.Dense(10, activation="softmax", name="output"),
        ],
        name="FCN",
    )

    # Compile model with optimizer
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

The model is now ready to be serialized and stored on the filesystem:

.. code-block:: python

    model_path, inputs, outputs = freeze_model(model, '.', "mnist.pb")

Note that ``freeze_model`` conveniently returns the path to the serialized model file,
and the names of the input and output layers, which are needed to upload the TensorFlow
model on the DB, as shown in the following code snippet, where we also upload a
synthetic sample to be passed to the model. Notice that we could also upload a batch
of samples, instead of a single one. For details about ``set_model_from_file``, please
refer to :ref:`SmartRedis API <smartredis-api>`.

.. code-block:: python

    client = Client(cluster=False)
    model_key = "tf_mnist"
    client.set_model_from_file(
        model_key, model_path, "TF", device="GPU", inputs=inputs, outputs=outputs
    )
    mnist_image = np.random.rand(1, 28, 28).astype(np.float32)
    client.put_tensor("mnist_input", mnist_image)

Finally, the model can be run on the sample and the output is ready to be downloaded.

.. code-block:: python

    client.run_model(model_key, "mnist_input", "mnist_output")
    pred = client.get_tensor("mnist_output")


If storing the model as a file is not needed, then it can just be kept in memory
after serialization, using ``serialize_model`` after compiling the model. The same
workflow we saw in the previous example can then basically be achieved by replacing
``set_model_from_file`` with ``set_model``:

.. code-block:: python

    # ... standard imports
    from smartsim.ml.tf import serialize_model

    # ... define, instantiate, and compile Keras model

    serialized_model, inputs, outputs = serialize_model(model)

    client = Client(cluster=False)
    model_key = "tf_mnist_serialized"
    client.set_model(
        model_key, serialized_model, "TF", device="GPU", inputs=inputs, outputs=outputs
    )
    mnist_image = np.random.rand(1, 28, 28).astype(np.float32)
    client.put_tensor("mnist_input", mnist_image)

    client.run_model(model_key, "mnist_input", "mnist_output_serialized")
    pred = client.get_tensor("mnist_output_serialized")


PyTorch
-------
PyTorch requires models to be `jit-traced <https://pytorch.org/docs/1.11/generated/torch.jit.save.html>`__.
The method ``torch.jit.save`` can either store the model in memory or on file.

First, we define the model and a

.. code-block:: python

    import io

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from smartredis import Client

    # simple MNIST in PyTorch
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output

We can then creat the Neural Network, jit-trace it and upload it
to the DB. Note that we are storing the serialized model in a ``BytesIO``
object, which means that we are keeping it in memory and not storing
it on the file system. For this reason, we need to call SmartRedis's
``Client.set_model()`` method.


.. code-block:: python

    n = Net()
    example_forward_input = torch.rand(20, 1, 28, 28)
    module = torch.jit.trace(n, example_forward_input)
    model_buffer = io.BytesIO()
    torch.jit.save(module, model_buffer)
    net = model_buffer.getvalue()

    # connect a client to the database
    client = Client(cluster=False)

    # 20 samples of "image" data
    client.set_model("cnn", net, "TORCH", device="CPU")
    client.put_tensor("input", example_forward_input.numpy())
    client.run_model("cnn", inputs=["input"], outputs=["output"])
    output = client.get_tensor("output")

We can also store the serialized model on the file system as follows.

.. code-block:: python

    n = Net()
    example_forward_input = torch.rand(20, 1, 28, 28)
    module = torch.jit.trace(n, example_forward_input)
    torch.jit.save(module, "traced_model.pt")

    # connect a client to the database
    client = Client(cluster=False)

    # 20 samples of "image" data
    client.set_model_from_file("cnn", "traced_model.pt", "TORCH", device="CPU")
    client.put_tensor("input", example_forward_input.numpy())
    client.run_model("cnn", inputs=["input"], outputs=["output"])
    output = client.get_tensor("output")


TorchScript Functions
---------------------
Instead of Neural Networks, or, in general, Machine Learning models, it is
possible to upload to the DB (collections of) functions which can be used e.g.
to perform pre- or post-processing operations on tensors stored on the DB.

Since the functions are going to be stored as TorchScript modules, they
- need to be jit-traceable
- can use ``torch`` as a built-in module
- can **not** import modules

The easiest way of defining and storing functions on the DB is to create a
dedicated file. In that file, we can define functions which will be callable
through the SmartRedis ``Client``, but also from other functions in the
same file. A typical script file would look like this:

.. code-block:: python

    def rescale(tensor, mu: float, sigma: float):
        mean = tensor.mean()
        std = tensor.std()

        normalized = (tensor-mean)/std
        return tensor*sigma + mu

    def shift_y_to_x(x, y):
        mu_x = x.mean()
        sigma_x = x.std()
        y_rescaled = rescale(y, mu_x, mu_y)

        return y_rescaled

In the script, we defined ``shift_y_to_x``,
a function which returns a modified copy of a tensor ``y``,
which matches the statistical distribution of the tensor ``x``.
Notice that we are not importing ``torch`` in the script, as it will
be recognized as a built-in by the TorchScript compiler.

Here is the code which allows us to run the function ``shift_y_to_x`` on
tensors stored in the DB. We will assume that the above script is stored
as ``"./shift.py"``.

.. code-block:: python

    import numpy as np
    from smartredis import Client

    x = np.random.rand(100, 100).astype(np.float32)
    y = np.random.rand(100, 100).astype(np.float32) * 2 + 10

    client = Client(cluster=False)
    client.put_tensor("X_rand", x)
    client.put_tensor("Y_rand", y)

    client.set_script_from_file("shifter", "./shift.py", device="CPU")
    client.run_script("shifter", "shift_y_to_x_points", inputs=["X_rand", "Y_rand"], outputs=["Y_scaled"])
    y_scaled = client.get_tensor("Y_scaled")

Simpler functions (or functions that do not require calling other functions),
can be defined inline and uploaded to the DB. For example:


.. code-block:: python

    import numpy as np
    from smartredis import Client

    def normalize(X):
        mean = X.mean()
        std = X.std()

        return (X-mean)/std

    x = np.random.rand(100, 100).astype(np.float32) * 2 + 10

    client = Client(cluster=False)
    client.put_tensor("X_rand", x)

    client.set_function("normalizer", normalize)
    client.run_script("normalizer", "normalize", inputs=["X_rand"], outputs=["X_norm"])
    x_norm = client.get_tensor("X_norm")

Notice that the key ``"normalizer"`` represents the script containing the function (similar to
``"shifter"`` in the previous example), while the function name is ``"normalize"``.

ONNX Runtime
------------

Thanks to the ONNX runtime, Machine Learning and Data Analysis functions defined in
Scikit-Learn can be used in the DB. In the following example, we see how a model
representing a linear regression can be uploaded to the DB and applied to a tensor.

.. code-block:: python

    import numpy as np
    from skl2onnx import to_onnx
    from sklearn.linear_model import LinearRegression
    from smartredis import Client

    def build_lin_reg():
        x = np.array([[1.0], [2.0], [6.0], [4.0], [3.0], [5.0]]).astype(np.float32)
        y = np.array([[2.0], [3.0], [7.0], [5.0], [4.0], [6.0]]).astype(np.float32)

        linreg = LinearRegression()
        linreg.fit(x, y)
        linreg = to_onnx(linreg, x.astype(np.float32), target_opset=13)
        return linreg.SerializeToString()

    # connect a client to the database
    client = Client(cluster=False)

    # linreg test
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]).astype(np.float32)
    linreg = build_lin_reg()
    outputs = run_model(client, "linreg", device, linreg, X, "X", ["Y"])
    run_model(client, model_name, device, model, model_input, in_name, out_names):
    client.put_tensor("X", X)
    client.set_model("linreg", linreg, "ONNX", device="GPU")
    client.run_model("linreg", inputs=["X"], outputs=["Y"])

    Y = client.get_tensor("Y")
