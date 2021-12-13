
================
Machine Learning
================

Compiling TensorFlow or PyTorch runtimes into each existing simulation is
difficult. Maintaining that type of integration with the rapidly growing and changing
APIs of libraries like TensorFlow and PyTorch is even more difficult.

Instead of forcing dependencies on the simulation code, SmartSim itself maintains those dependencies
and provides simulations runtime access to them through the ``Orchestrator`` database.

Simulations in Fortran, C, C++ and Python can call into PyTorch, TensorFlow,
and any library that supports the ONNX format without compiling in ML libraries into the
simulation.

.. note::

    While the SmartRedis examples below are written in Python, SmartRedis is implemented
    in C, C++ and Fortran as well. Fortran, C, and C++ applications can all call the
    same Machine Learning libraries/models as the examples below.


4.1 Your First Inference Session
================================

.. _infrastructure_code:

SmartSim performs online inference by using the SmartRedis clients to call into the
Machine Learning runtimes linked into the Orchestrator database.

Therefore, to perform inference, you must first create an Orchestrator database and
launch it. The code below can be used to launch a database with SmartSim and Python
script that uses the SmartRedis Python client to perform innovations of the ML runtimes.


.. code-block:: python

    from smartsim import Experiment
    from smartsim.database import Orchestrator
    from smartsim.settings import RunSettings

    exp = Experiment("inference-session", launcher="local")
    db = Orchestrator(port=6780)

    script = "inference.py"
    settings = RunSettings("python", exe_args=script)
    model = exp.create_model("model_using_ml", settings)
    model.attach_generator_files(to_copy=script)

    exp.start(db, model, block=True, summary=True)
    exp.stop(db)

The above script will first launch the database, and then the script
containing the SmartRedis client code Python script. The code here could
easily be adapted to launch a C, C++, or Fortran application containing
the SmartRedis clients in those languages as well.

Below are a few examples of scripts that could be used with the above
code to perform online inference with various ML backends supported
by SmartSim.


.. note::
    Online inference is not online training.
    The following code examples do not include code to train the models shown.


4.1.1 PyTorch
-------------

.. _TorchScript: https://pytorch.org/docs/stable/jit.html
.. _PyTorch: https://pytorch.org/
.. _trace: https://pytorch.org/docs/stable/generated/torch.jit.trace.html#torch.jit.trace

The Orchestrator supports both `PyTorch`_ models and `TorchScript`_ functions and scripts
in `PyTorch`_ 1.7.1. To use ONNX in SmartSim, specify
``TORCH`` as the argument for *backend* in the call to ``client.set_model`` or
``client.set_model_from_file``.

The below script can be used with the :ref:`SmartSim code <infrastructure_code>`
above to launch an inference session with a PyTorch model.

First, a PyTorch model is defined.

.. code-block:: python

    import io
    import numpy as np

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from smartredis import Client

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


Next we create a function to "jit-trace" the model and save it to a buffer.
If you aren't familier with the concept of tracing, take a look at the
Torch documentation for `trace`_.

.. code-block:: python

    n = Net()
    example_forward_input = torch.rand(1, 1, 28, 28)

    def create_torch_model(torch_module, example_forward_input):

        # perform the trace of the nn.Module.forward() method
        module = torch.jit.trace(torch_module, example_forward_input)

        # save the traced module to a buffer
        model_buffer = io.BytesIO()
        torch.jit.save(module, model_buffer)
        return model_buffer.getvalue()

Lastly, we use the SmartRedis Python client to
  1. Connect to the database
  2. Put a batch of 20 tensors into the database  (``put_tensor``)
  3. Set the Torch model in the database (``set_model``)
  4. Run the model on the batch of tensors (``run_model``)
  5. Retrieve the result (``get_tensor``)


.. code-block:: python

    client = Client(cluster=False)

    client.put_tensor("input", torch.rand(20, 1, 28, 28).numpy())

    # put the PyTorch CNN in the database in GPU memory
    client.set_model("cnn", net, "TORCH", device="GPU")

    # execute the model, supports a variable number of inputs and outputs
    client.run_model("cnn", inputs=["input"], outputs=["output"])

    # get the output
    output = client.get_tensor("output")
    print(f"Prediction: {output}")

Since we are launching the inference
script through SmartSim, we do not need to specify the address of the
database as SmartSim will connect the Client for us. Additionally,
``cluster=False`` is specified so the client will not attempt to find
other cluster shards on the network.

If running on CPU, be sure to change the argument in the ``set_model`` call
above to ``CPU``.


4.1.2 TensorFlow and Keras
--------------------------

.. _TensorFlow: https://www.tensorflow.org/
.. _Keras: https://keras.io/

The Orchestrator, in addition to PyTorch, is built with `TensorFlow`_ and `Keras`_ support by default.
Currently TensorFlow 2.4.2 is supported, but the graph of the model must be frozen
before it is placed in the database. This is the same process for both Keras and
TensorFlow.

The example below shows how to prepare a simple Keras model for use with SmartSim.
This script can be used with the :ref:`SmartSim code <infrastructure_code>`
above to launch an inference session with a TensorFlow or Keras model.

First, a simple Keras Convolutional Neural Network is defined.

.. code-block:: python

    import os
    import numpy as np
    from tensorflow import keras


    # create a simple Fully connected network in Keras
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
    model.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])


After a model is created (trained or not), the graph of the model is
frozen saved to file so the client method ``client.set_model_from_file``
can load it into the database.

SmartSim includes a utility to freeze the graph of a TensorFlow or Keras model in
:ref:`smartsim.tf <smartsim_tf_api>`. To use TensorFlow or Keras in SmartSim, specify
``TF`` as the argument for *backend* in the call to ``client.set_model`` or
``client.set_model_from_file``.

Note that TensorFlow and Keras, unlike the other ML libraries supported by
SmartSim, requires an ``input`` and ``output`` argument in the call to
``set_model``. These arguments correspond to the layer names of the
created model. The :ref:`smartsim.tf.freeze_model <smartsim_tf_api>` utility
returns these values for convenience as shown below.

.. code-block:: python

    from smartredis import Client
    from smartsim.tf import freeze_model


    # SmartSim utility for Freezing the model
    model_path, inputs, outputs = freeze_model(model, os.getcwd(), "fcn.pb")

    client = Client(cluster=False)

    # TensorFlow backed requires named inputs and outputs on graph
    # this differs from PyTorch and ONNX.
    client.set_model_from_file(
        "keras_fcn", model_path, "TF", device=device, inputs=inputs, outputs=outputs
    )

    input_data = np.random.rand(1, 28, 28).astype(np.float32)
    client.put_tensor("input", input_data)
    client.run_model("keras_fcn", "input", "output")

    pred = client.get_tensor("output")
    print(pred)


4.1.3 ONNX
----------

.. _Scikit-learn: https://scikit-learn.org
.. _XGBoost: https://xgboost.readthedocs.io
.. _CatBoost: https://catboost.ai
.. _LightGBM: https://lightgbm.readthedocs.io/en/latest/
.. _libsvm: https://www.csie.ntu.edu.tw/~cjlin/libsvm/

.. _onnxmltools: https://github.com/onnx/onnxmltools
.. _skl2onnx: https://github.com/onnx/sklearn-onnx/
.. _tensorflow-onnx: https://github.com/onnx/tensorflow-onnx/

ONNX is a standard format for representing models. A number of different Machine Learning
Libraries are supported by ONNX and can be readily used with SmartSim.

Some popular ones are:

- `Scikit-learn`_
- `XGBoost`_
- `CatBoost`_
- `TensorFlow`_
- `Keras`_
- `PyTorch`_
- `LightGBM`_
- `libsvm`_

As well as some that are not listed. There are also many tools to help convert
models to ONNX.

- `onnxmltools`_
- `skl2onnx`_
- `tensorflow-onnx`_

And PyTorch has it's own converter.

Below are some examples of a few models in `Scikit-learn`_ that are converted
into ONNX format for use with SmartSim. To use ONNX in SmartSim, specify
``ONNX`` as the argument for *backend* in the call to ``client.set_model`` or
``client.set_model_from_file``.

These scripts can be used with the :ref:`SmartSim code <infrastructure_code>`
above to launch an inference session with any of the supported ONNX libraries.

KMeans
++++++

.. _skl2onnx.to_onnx: http://onnx.ai/sklearn-onnx/auto_examples/plot_convert_syntax.html

K-means clustering is an unsupervised ML algorithm. It is used to categorize data points
into f groups ("clusters"). Scikit Learn has a built in implementation of K-means clustering
and it is easily converted to ONNX for use with SmartSim through `skl2onnx.to_onnx`_.

Since the KMeans model returns two outputs, we provide the ``client.run_model`` call
with two ``outputs``.

.. code-block:: python

    X = np.arange(20, dtype=np.float32).reshape(10, 2)
    tr = KMeans(n_clusters=2)
    tr.fit(X)

    kmeans = to_onnx(tr, X, target_opset=11)
    model = kmeans.SerializeToString()

    sample = np.arange(20, dtype=np.float32).reshape(10, 2) # dummy data
    client.put_tensor("input", sample)

    client.set_model("kmeans", model, "ONNX", device="CPU")
    client.run_model("kmeans", inputs="input", outputs=["labels", "transform"])

    print(client.get_tensor("labels"))


Random Forest
+++++++++++++

The Random Forest example uses the Iris dataset from Scikit Learn to train a
RandomForestRegressor. As with the other examples, the skl2onnx function
`skl2onnx.to_onnx`_ is used to convert the model to ONNX format.

.. code-block:: python

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, _ = train_test_split(X, y, random_state=13)
    clr = RandomForestRegressor(n_jobs=1, n_estimators=100)
    clr.fit(X_train, y_train)

    rf_model = to_onnx(clr, X_test.astype(np.float32))

    sample = np.array([[6.4, 2.8, 5.6, 2.2]]).astype(np.float32)
    model = rf_model.SerializeToString()

    client.put_tensor("input", sample)
    client.set_model("rf_regressor", model, "ONNX", device="CPU")
    client.run_model("rf_regressor", inputs="input", outputs="output")
    print(client.get_tensor("output"))


4.2 Online training with SmartSim
=================================

A SmartSim ``Orchestrator`` can be used to store and retrieve samples and targets used to
train a ML model. A typical example is one in which one simulation produces samples at
each time step and another application needs to download the samples as they are produced 
to train a Deep Neural Network (e.g. a surrogate model).

In this section, we will use components implemented in ``smartsim.ml.tf.data``, to train a
Neural Network implemented in TensorFlow and Keras. In particular, we will be using
two classes:
- ``smartsim.ml.data.TrainingUploader`` which streamlines the uploading of samples and corresponding targets to the DB
- ``smartsim.ml.tf.data.DataGenerator`` which is a Keras ``Generator`` which can be used to train a DNN,
and will download the samples from the DB updating the training set at the end of each epoch.

The SmartSim ``Experiment`` will consist in one mock simulation (the ``producer``) uploading samples,
and one application (the ``training_service``) downloading the samples to train a DNN.

A richer example, entirely implemented in Python, is available as a Jupyter Notebook in the
``tutorials`` section of the SmartSim repository.
An equivalent example using PyTorch instead of TensorFlow is available in the same directory.


4.2.1 Producing and uploading the samples
-----------------------------------------

.. _ml_training_producer_code:

The first application in the workflow, the ``producer`` will upload batches of samples at regular intervals,
mimicking the behavior of an iterative simulation. 

Since the ``training_service`` will use a ``smartsim.ml.tf.DataGenerator`` two download the samples, their
keys need to follow a pre-defined format. Assuming that only one process in the simulation
uploads the data, this format is ``<sample_prefix>_<iteration>``. And for targets
(which can also be integer labels), the key format is ``<target_prefix>_<iteration>``. Both ``<sample_prefix>``
and ``<target_prefix>`` are user-defined, and will need to be used to initialize the
``smartsim.ml.tf.DataGenerator`` object.

Assuming the simulation is written in Python, then the code would look like

.. code-block:: python

    from SmartRedis import Client
    # simulation initialization code
    client = Client(cluster=False, address=None)

    for iteration in range(num_iterations):
        # simulation code producing two tensors, data_points
        # and data_values
        client.put_tensor(f"points_{iteration}", data_points)
        client.put_tensor(f"values_{iteration}", data_values)


For simple simulations, this is sufficient. But if the simulation
uses MPI, then each rank could upload a portion of the data set. In that case,
the format for sample and target keys will be ``<sample_prefix>_<sub-index>_<iteration>``
and ``<target_prefix>_<sub-index>_<iteration>``, where ``<sub_index>`` can be, e.g.
the MPI rank id.


4.2.1 Downloading the samples and training the model
----------------------------------------------------

The second part of the workflow is the ``training_service``, an application that
downloads the data uploaded by the ``producer`` and uses them to train a ML model.
Most importantly, the ``training_service`` needs to keep looking for new samples,
and download them as they are available. The training data set size thus needs to grow at
each ``producer`` iteration.

In Keras, a ``Sequence`` represents a data set and can be passed to ``model.fit()``.
The class ``smartsim.ml.tf.DataGenerator`` is a Keras ``Sequence``, which updates
its data set at the end of each training epoch, looking for newly produced batches of samples.
A current limitation of the TensorFlow training algorithm is that it does not take
into account changes of size in the data sets once the training has started, i.e. it is always
assumed that the training (and validation) data does not change during the training. To
overcome this limitation, we need to train one epoch at the time. Thus, 
following what we defined in the :ref:`producer section <ml_training_produced_code>`,
the ``training_service`` would look like

.. code-block:: python

    from smartsim.ml.tf.data import DataGenerator
    generator = DataGenerator(
        sample_prefix="points",
        target_prefix="value",
        batch_size=32,
        smartredis_cluster=False)

    model = # some ML model
    # model initialization

    for epoch in range(100):
        model.fit(generator,
                  steps_per_epoch=None, 
                  epochs=epoch+1,
                  initial_epoch=epoch, 
                  batch_size=generator.batch_size,
                  verbose=2)


Again, this is enough for simple simulations. If the simulation uses MPI,
then the ``DataGenerator`` needs to know about the possible sub-indices. For example,
if the simulation runs 8 MPI ranks, the ``DataGenerator`` initialization will
need to be adapted as follows

.. code-block:: python

    generator = DataGenerator(
        sample_prefix="points",
        target_prefix="value",
        batch_size=32,
        smartredis_cluster=False,
        sub_indices=8)


4.2.2 Launching the experiment
------------------------------

To launch the ``producer`` and the ``training_service`` as models
within a SmartSim ``Experiment``, we can use the following code:

.. code-block:: python
    
    from smartsim import Experiment
    from smartsim.database import Orchestrator
    from smartsim.settings import RunSettings

    db = Orchestrator(port=6780)
    exp = ("online-training", launcher="local")

    # producer
    producer_script = "producer.py"
    settings = RunSettings("python", exe_args=producer_script)
    uploader_model = exp.create_model("producer", settings)
    uploader_model.attach_generator_files(to_copy=script)
    uploader_model.enable_key_prefixing()

    # training_service
    training_script = "training_service.py"
    settings = RunSettings("python", exe_args=training_script)
    trainer_model = exp.create_model("training_service", settings)
    trainer_model.register_incoming_entity(uploader_model)

    exp.start(db)
    exp.start(uploader_model, block=False, summary=False)
    exp.start(trainer_model, block=True, summary=False)


Two lines require attention, as they are needed by the ``DataGenerator`` to work:
- ``uploader_model.enable_key_prefixing()`` will ensure that the ``producer`` prefixes
all tensor keys with its name
- ``trainer_model.register_incoming_entity(uploader_model)`` enables the ``DataGenerator``
in the ``training_service`` to know that it needs to download samples produced by the ``producer``


