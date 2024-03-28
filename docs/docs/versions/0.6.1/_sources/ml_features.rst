###########
ML Features
###########

In this section, we illustrate features which
users are expected to use in HPC workloads, especially when
simulation and AI are required to interact. The topics are
explained through code snippets,
with code that goes beyond SmartSim and SmartRedis API
(e.g. code showing how to jit-script a PyTorch model): the
intention is that of showing *one* simple way of leveraging
a feature, but more optimized ways of using third-party
libraries may exist.

Examples are written in Python, but the same
result can be achieved with any SmartRedis client (C, C++,
Fortran and Python). Please refer to SmartRedis API
for language-specific details.

ML Model Deployment and Execution in the Database
===================================================

The combination of SmartSim and SmartRedis enables users
to store more than simple tensors on the database (DB).
In the upcoming subsections, we demonstrate how to use a
SmartRedis client to upload executable code, in the
form of ML model, scripts, and functions, to the DB.
Once store, the code can be executed using the SmartRedis client
methods and used to process tensors directly in the DB.
The tensors generated from running the stored code will also be stored
in the database and can be retrieved with standard SmartRedis ``Client.get_tensor()`` calls.

SmartRedis offers two ways to upload serialized code
to the DB: from memory and from file. We will go through examples
demonstrating how to upload from each. We provide the following examples:

- :ref:`TensorFlow and PyTorch <ml_features_TF_PT>`: Serialize a TensorFlow/Keras or PyTorch model, optionally
  save it to file, upload it to the DB, then execute it on tensors stored on the DB.
- :ref:`TorchScript Functions <ml_features_torchscript>`: Serialize TorchScript functions, optionally
  save them to file, upload them to the DB, then execute them on tensors stored on the DB.
- :ref:`ONNX Runtime <ml_features_onnx>`: Convert a Scikit-Learn model to ONNX
  format, upload it to the DB, then execute it on tensors stored on the DB.


.. note::
    In all examples, we will assume that a SmartSim ``Orchestator``
    is up and running, and that the code we will show is run as part
    of a SmartSim-launched application ``Model``.


.. _ml_features_TF_PT:

TensorFlow and PyTorch
----------------------

In this section, we will see how a TensorFlow/Keras or a PyTorch model
can be serialized using SmartSim's helper functions.
Once the model is serialized, we will use the SmartRedis client to upload it to the DB,
and execute it on data stored on the DB.
We will also see how the model can be optionally saved to file. The
workflow for TensorFlow and PyTorch is almost identical, but we provide
the code for each toolkit in a dedicated tab, for completeness.

We begin by defining the ML model that we will use in both examples of
this section.

.. tabs::

    .. group-tab:: TensorFlow

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

    .. group-tab:: PyTorch

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

            # Instantiate the model
            n = Net()


============================================================
Serializing the model and uploading it to the DB from memory
============================================================
Once the model is instantiated, it needs to be serialized to be uploaded
to the DB using the SmartRedis client.

.. tabs::

    .. group-tab:: TensorFlow

        As part of its :ref:`TensorFlow helper functions <smartsim_tf_api>`,
        SmartSim provides ``serialize_model()`` to serialize a TensorFlow or Keras
        model.

        .. code-block:: python

            serialized_model, inputs, outputs = serialize_model(model)


        Note that ``serialize_model()`` conveniently returns the model as bytestring
        and the names of the input and output layers, which are now needed to upload the TensorFlow
        model to the DB using ``Client.set_model()``.
        We also use ``Client.put_tensor()`` to upload a batch of 20 synthetic MNIST samples to the DB.

        .. code-block:: python

            # Instantiate and connect SmartRedis client to communicate with DB
            client = Client(cluster=False)
            model_key = "mnist_cnn"
            # Set device to CPU if GPU not available to DB
            client.set_model(
                model_key, serialized_model, "TF", device="GPU", inputs=inputs, outputs=outputs
            )


    .. group-tab:: PyTorch

        PyTorch requires models to be `jit-traced <https://pytorch.org/docs/2.0/generated/torch.jit.save.html>`__.
        The method ``torch.jit.save()`` can either store the model in memory or on file. Here,
        we will keep it in memory as a bytestring.

        .. code-block:: python

            # Example input needed for jit tracing
            example_forward_input = torch.rand(20, 1, 28, 28)
            module = torch.jit.trace(n, mnist_images)
            model_buffer = io.BytesIO()
            torch.jit.save(module, model_buffer)
            serialized_model = model_buffer.getvalue()

        Now that we have the serialized model, we can upload it to the DB using the ``Client.set_model()``.


        We also use ``Client.put_tensor()`` to upload a batch of 20 synthetic MNIST samples to the DB.

        .. code-block:: python

            # Instantiate and connect SmartRedis client to communicate with DB
            client = Client(cluster=False)
            model_key = "mnist_cnn"
            # Set device to CPU if GPU not available to DB
            client.set_model(model_key, serialized_model, "TORCH", device="GPU")


For details about ``Client.set_model()``, please
refer to :ref:`SmartRedis API <smartredis-api>`.


=====================================================
Saving the model to a file and uploading it to the DB
=====================================================

Once the model is compiled, it can be serialized and stored on the filesystem. This is
useful if the model has to be used at a later time. Once the model is saved to file,
it can be uploaded to the DB using the SmartRedis client.

.. tabs::

    .. group-tab:: TensorFlow

        As part of its :ref:`TensorFlow helper functions <smartsim_tf_api>`,
        SmartSim provides ``freeze_model()`` to serialize a TensorFlow or Keras
        model and save it to file. In this example, the file will be named ``mnist.pb``.

        .. code-block:: python

            filename = "mnist.pb"
            model_path, inputs, outputs = freeze_model(model, '.', filename)


        Note that ``freeze_model()`` conveniently returns the path to the serialized model file,
        and the names of the input and output layers, which are noew needed to upload the TensorFlow
        model to the DB using ``Client.set_model_from_file()``. We also use
        ``Client.put_tensor()`` to upload a synthetic MNIST sample to the DB.


        .. code-block:: python

            client = Client(cluster=False)
            model_key = "mnist_cnn"
            client.set_model_from_file(
                model_key, model_path, "TF", device="GPU", inputs=inputs, outputs=outputs
            )


    .. group-tab:: PyTorch

        PyTorch requires models to be `jit-traced <https://pytorch.org/docs/2.0/generated/torch.jit.save.html>`__.
        The method ``torch.jit.save()`` can either store the model in memory or on file. Here,
        we will save it to a file located at ``./traced_model.pt``.

        .. code-block:: python

            # Example input needed for jit tracing
            example_forward_input = torch.rand(20, 1, 28, 28)
            module = torch.jit.trace(n, example_forward_input)
            model_path = "./traced_model.pt"
            torch.jit.save(module, modelpath)


        Now that we have the serialized model, we can upload it to the DB using
        ``Client.set_model_from_file()`` method.

        .. code-block:: python

            client = Client(cluster=False)
            model_key = "mnist_cnn"

            client.set_model_from_file(model_key, model_path, "TORCH", device="CPU")


For details about ``Client.set_model_from_file()``, please
refer to :ref:`SmartRedis API <smartredis-api>`.

===============================================
Executing the model on tensors stored in the DB
===============================================

Now that the model is available for execution on the DB, we use the SmartRedis client
to upload a tensor representing a batch of 20 synthetic MNIST images.

.. tabs::

    .. group-tab:: TensorFlow

        .. code-block:: python

            # 20 samples of "image" data
            mnist_images = np.random.rand(20, 28, 28, 1).astype(np.float32)
            # client was instantiated previously
            client.put_tensor("mnist_images", mnist_image)


    .. group-tab:: PyTorch

        .. code-block:: python


            # 20 samples of "image" data
            mnist_images = torch.rand(20, 1, 28, 28)
            # client was instantiated previously
            client.put_tensor("mnist_images", mnist_images.numpy())


Now we can use ``Client.run_model()`` to execute the model on the data we have
just stored and ``Client.get_tensor()`` to download the output of the model execution.
Notice that, for this part, the code is identical for models uploaded from file and from memory, and
with TensorFlow or PyTorch backends.

.. code-block:: python

    client.run_model(model_key, inputs=["mnist_imagse"], outputs=["mnist_output"])
    output = client.get_tensor("mnist_output")


For details about ``Client.run_model()``, please
refer to :ref:`SmartRedis API <smartredis-api>`.

.. _ml_features_torchscript:

TorchScript Functions
---------------------
Instead of Neural Networks, or, in general, Machine Learning models, it is
possible to upload to the DB (collections of) functions which can be used e.g.
to perform pre- or post-processing operations on tensors stored on the DB.

Since the functions are going to be stored as TorchScript modules, they

- need to be jit-traceable
- can use ``torch`` as a built-in module
- can **not** import modules

In this section we will see how to

- save a collection of functions to a script file, upload them to the DB,
  and execute them on tensors stored on the DB.
- define and upload a function on-the-fly from a Python script and
  execute it on tensors stored on the DB.


=================================================================
Uploading a script containing a collection of functions to the DB
=================================================================

The easiest way of defining and storing functions on the DB is to create a
dedicated file. In that file, we can define functions which will be callable
through the SmartRedis client, but also from other functions in the
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
        y_rescaled = rescale(y, mu_x, sigma_x)

        return y_rescaled

In the script, we defined ``shift_y_to_x``,
a function which returns a modified copy of a tensor ``y``,
which matches the statistical distribution of the tensor ``x``.
Notice that we are not importing ``torch`` in the script, as it will
be recognized as a built-in by the TorchScript compiler. Because
of the discrepancy between TorchScript's and Python's syntaxes, TorchScript
scripts cannot be run as standalone Python scripts.

Here is the code which allows us to run the function ``shift_y_to_x`` on
tensors stored in the DB. We will assume that the above script is stored
as ``"./shift.script"``.

.. code-block:: python

    import numpy as np
    from smartredis import Client

    # Generate tensors according to two different random distributions
    x = np.random.rand(100, 100).astype(np.float32)
    y = np.random.rand(100, 100).astype(np.float32) * 2 + 10

    # Instantiate and connect SmartRedis client
    client = Client(cluster=False)

    # Upload tensors to DB
    client.put_tensor("X_rand", x)
    client.put_tensor("Y_rand", y)

    # Upload script containing functions to DB
    client.set_script_from_file("shifter", "./shift.script", device="CPU")
    # Run the function ``shift_y_to_x`` on ``X_rand`` and ``Y_rand``
    client.run_script("shifter", "shift_y_to_x", inputs=["X_rand", "Y_rand"], outputs=["Y_scaled"])
    # Download output
    y_scaled = client.get_tensor("Y_scaled")


In the above code, we used ``Client.put_tensor()`` to upload tensors to the DB, and
``Client.set_script_from_file()`` to upload the script containing the collection of functions.
We then used ``Client.run_script()`` to run the function ``shift_y_to_x`` on the stored
tensors, and downloaded the result with ``Client.get_tensor()``.

For details about ``Client.set_script_from_file()`` and ``Client.run_script()``, please
refer to :ref:`SmartRedis API <smartredis-api>`.


=========================================
Uploading a function to the DB on-the-fly
=========================================

Simpler functions (or functions that do not require calling other user-defined
or imported functions), can be defined inline and uploaded to the DB using the SmartRedis client.
For example:

.. code-block:: python

    import numpy as np
    from smartredis import Client

    def normalize(X):
        """Simple function to normalize a tensor"""
        mean = X.mean()
        std = X.std()

        return (X-mean)/std

    # Generate random tensor
    x = np.random.rand(100, 100).astype(np.float32) * 2 + 10

    # Instantiate and connect SmartRedis client
    client = Client(cluster=False)

    # Upload tensor to DB
    client.put_tensor("X_rand", x)

    # Upload function to DB, ``normalizer`` is the name of the collection
    # of functions containing the function ``normalize`` only. It mimics
    # the way `set_script` works.
    client.set_function("normalizer", normalize)
    # Run the function ``normalize`` on ``X_rand``
    client.run_script("normalizer", "normalize", inputs=["X_rand"], outputs=["X_norm"])
    # Download output
    x_norm = client.get_tensor("X_norm")

Notice that the key ``"normalizer"`` represents the script containing the function (similar to
``"shifter"`` in the previous example), while the function name is ``"normalize"``.


For details about ``Client.set_function()`` and  ``Client.run_script()``, please
refer to :ref:`SmartRedis API <smartredis-api>`.

.. _ml_features_ONNX:

ONNX Runtime
------------

In the following example, we will see how, thanks to the ONNX runtime,
Machine Learning and Data Analysis functions defined in
Scikit-Learn can be serialized and then put on the DB using the SmartRedis client.

We start by defining a Scikit-Learn ``LinearRegression`` model and serialize it,
keeping it into memory.

.. code-block:: python

    import numpy as np
    from skl2onnx import to_onnx
    from sklearn.linear_model import LinearRegression
    from smartredis import Client

    def build_lin_reg():
        """Generates sklearn linear regression model and serialize it"""
        x = np.array([[1.0], [2.0], [6.0], [4.0], [3.0], [5.0]]).astype(np.float32)
        y = np.array([[2.0], [3.0], [7.0], [5.0], [4.0], [6.0]]).astype(np.float32)

        linreg = LinearRegression()
        linreg.fit(x, y)
        linreg = to_onnx(linreg, x.astype(np.float32), target_opset=13)
        return linreg.SerializeToString()

    linreg = build_lin_reg()

Once the model is serialized, we can use ``Client.set_model()`` to upload it
to the DB.

.. code-block:: python

    # connect a client to the database
    client = Client(cluster=False)
    client.set_model("linreg", linreg, "ONNX", device="GPU")


Finally, we can upload a tensor to the DB using ``Client.put_tensor()``, run the
stored model on it using ``Client.run_model()``, and download the output calling
``Client.get_tensor()``.

.. code-block:: python

    # linreg test
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]).astype(np.float32)
    client.put_tensor("X", X)
    client.run_model("linreg", inputs=["X"], outputs=["Y"])
    Y = client.get_tensor("Y")


For details about ``Client.run_model()``, please
refer to :ref:`SmartRedis API <smartredis-api>`.
