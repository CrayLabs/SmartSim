
===============
Online Training
===============

Online training provides the ability to use dynamic processes as your training
data set. In SmartSim, training data can be any process using the SmartRedis clients
to store data inside of a deployed `Orchestrator` database.

SmartSim includes utilizes to help with online training workflows in PyTorch and TensorFlow
In this example, we show how to use ``smartsim.ml.tf`` to train a Neural Network implemented
in TensorFlow and Keras.

In particular, we will be using two classes:
- ``smartsim.ml.data.TrainingUploader`` which streamlines the uploading of samples and corresponding targets to the DB
- ``smartsim.ml.tf.DataGenerator`` which is a Keras ``Generator`` which can be used to train a DNN,
and will download the samples from the DB updating the training set at the end of each epoch.

The SmartSim ``Experiment`` will consist in one mock simulation (the ``producer``) uploading samples,
and one application (the ``training_service``) downloading the samples to train a DNN.

A richer example, entirely implemented in Python, is available as a Jupyter Notebook in the
``tutorials`` section of the SmartSim repository. An equivalent example using PyTorch
instead of TensorFlow is available in the same directory.


Producing and uploading the samples
-----------------------------------

.. _ml_training_producer_code:

The first application in the workflow, the ``producer`` will upload batches of samples at regular intervals,
mimicking the behavior of an iterative simulation.

Since the ``training_service`` will use a ``smartsim.ml.tf.DynamicDataGenerator`` two download the samples, their
keys need to follow a pre-defined format. Assuming that only one process in the simulation
uploads the data, this format is ``<sample_prefix>_<iteration>``. And for targets
(which can also be integer labels), the key format is ``<target_prefix>_<iteration>``. Both ``<sample_prefix>``
and ``<target_prefix>`` are user-defined, and will need to be used to initialize the
``smartsim.ml.tf.DynamicDataGenerator`` object.

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


Downloading the samples and training the model
----------------------------------------------

The second part of the workflow is the ``training_service``, an application that
downloads the data uploaded by the ``producer`` and uses them to train a ML model.
Most importantly, the ``training_service`` needs to keep looking for new samples,
and download them as they are available. The training data set size thus needs to grow at
each ``producer`` iteration.

In Keras, a ``Sequence`` represents a data set and can be passed to ``model.fit()``.
The class ``smartsim.ml.tf.DynamicDataGenerator`` is a Keras ``Sequence``, which updates
its data set at the end of each training epoch, looking for newly produced batches of samples.
A current limitation of the TensorFlow training algorithm is that it does not take
into account changes of size in the data sets once the training has started, i.e. it is always
assumed that the training (and validation) data does not change during the training. To
overcome this limitation, we need to train one epoch at the time. Thus,
following what we defined in the :ref:`producer section <ml_training_producer_code>`,
the ``training_service`` would look like

.. code-block:: python

    from smartsim.ml.tf import DynamicDataGenerator
    generator = DynamicDataGenerator(
            sample_prefix="points",
            target_prefix="value",
            batch_size=32,
            cluster=False)

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
then the ``DynamicDataGenerator`` needs to know about the possible sub-indices. For example,
if the simulation runs 8 MPI ranks, the ``DynamicDataGenerator`` initialization will
need to be adapted as follows

.. code-block:: python

    generator = DynamicDataGenerator(
                    sample_prefix="points",
                    target_prefix="value",
                    batch_size=32,
                    cluster=False,
                    uploader_ranks=8)


Launching the experiment
------------------------

To launch the ``producer`` and the ``training_service`` as models
within a SmartSim ``Experiment``, we can use the following code:

.. code-block:: python

    from smartsim import Experiment
    from smartsim.database import Orchestrator

    db = Orchestrator(port=6780)
    exp = Experiment("online-training", launcher="local")

    # producer
    producer_script = "producer.py"
    settings = exp.create_run_settings("python", exe_args=producer_script)
    uploader_model = exp.create_model("producer", settings, enable_key_prefixing=True)
    uploader_model.attach_generator_files(to_copy=producer_script)

    # training_service
    training_script = "training_service.py"
    settings = exp.create_run_settings("python", exe_args=training_script)
    trainer_model = exp.create_model("training_service", settings)
    trainer_model.register_incoming_entity(uploader_model)

    exp.start(db, uploader_model, block=False, summary=False)
    exp.start(trainer_model, block=True, summary=False)


Two lines require attention, as they are needed by the ``DataGenerator`` to work:
  - ``uploader_model.enable_key_prefixing()`` will ensure that the ``producer`` prefixes all tensor keys with its name
  - ``trainer_model.register_incoming_entity(uploader_model)`` enables the ``DataGenerator`` in
    the ``training_service`` to know that it needs to download samples produced by the ``producer``

