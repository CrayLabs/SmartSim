.. _fortran_client_examples:

*******
Fortran
*******




In this section, examples are presented using the SmartRedis Fortran
API to interact with the RedisAI tensor, model, and script
data types.  Additionally, an example of utilizing the
SmartRedis ``DataSet`` API is also provided.

.. note::
      The Fortran API examples rely on the ``SSDB`` environment
      variable being set to the address and port of the Redis database.

.. note::
      The Fortran API examples are written
      to connect to a clustered database or clustered SmartSim Orchestrator.
      Update the ``Client`` constructor ``cluster`` flag to `.false.`
      to connect to a single shard (single compute host) database.

Error handling
==============

The core of the SmartRedis library is written in C++ which utilizes the
exception handling features of the language to catch errors. This same
functionality does not exist in Fortran, so instead most SmartRedis
methods are functions that return error codes that can be checked. This
also has the added benefit that Fortran programs can incorporate
SmartRedis calls within their own error handling methods. A full list of
return codes for Fortran can be found in ``enum_fortran.inc.`` Additionally, the
``errors`` module has ``get_last_error`` and ``print_last_error`` to retrieve
the text of the error message emitted within the C++ code.

Tensors
=======

The SmartRedis Fortran client is used to communicate between
a Fortran client and the Redis database. In this example,
the client will be used to send an array to the database
and then unpack the data into another Fortran array.

This example will go step-by-step through the program and
then present the entirety of the example code at the end.

**Importing and declaring the SmartRedis client**

The SmartRedis client must be declared as the derived type
``client_type`` imported from the ``smartredis_client`` module.

.. code-block:: fortran

  program example
    use smartredis_client, only : client_type

    type(client_type) :: client
  end program example

**Initializing the SmartRedis client**

The SmartRedis client needs to be initialized before it can be used
to interact with the database. Within Fortran this is
done by calling the type-bound procedure
``initialize`` with the input argument ``.true.``
if using a clustered database or ``.false.`` otherwise.

.. code-block:: fortran

  program example
    use smartredis_client, only : client_type

    type(client_type) :: client
    integer :: return_code

    return_code = client%initialize(.false.) ! Change .false. to true if using a clustered database
    if (return_code .ne. SRNoError) stop 'Error in initializing client'
  end program example

**Putting a Fortran array into the database**

After the SmartRedis client has been initialized,
a Fortran array of any dimension and shape
and with a type of either 8, 16, 32, 64 bit
``integer`` or 32 or 64-bit ``real`` can be
put into the database using the type-bound
procedure ``put_tensor``.
In this example, as a proxy for model-generated
data, the array ``send_array_real_64`` will be
filled with random numbers and stored in the
database using ``put_tensor``. This subroutine
requires the user to specify a string used as the
'key' (here: ``send_array``) identifying the tensor
in the database, the array to be stored, and the
shape of the array.

.. literalinclude:: ../smartredis/examples/serial/fortran/smartredis_put_get_3D.F90
  :linenos:
  :language: fortran
  :lines: 46-54

**Unpacking an array stored in the database**

'Unpacking' an array in SmartRedis refers to filling
a Fortran array with the values of a tensor
stored in the database.  The dimensions and type of
data of the incoming array and the pre-declared
array are checked within the client to
ensure that they match. Unpacking requires
declaring an array and using the ``unpack_tensor``
procedure.  This example generates an array
of random numbers, puts that into the database,
and retrieves the values from the database
into a different array.

.. literalinclude:: ../smartredis/examples/serial/fortran/smartredis_put_get_3D.F90
  :linenos:
  :language: fortran


Datasets
========

The following code snippet shows how to use the Fortran
Client to store and retrieve dataset tensors and
dataset metadata scalars.

.. literalinclude:: ../smartredis/examples/serial/fortran/smartredis_dataset.F90
  :linenos:
  :language: fortran

.. _SR Fortran Models:

Models
======

For an example of placing a model in the database
and executing the model using a stored tensor,
see the :ref:`SR Parallel MPI` example.  The
aforementioned example is customized to show how
key collisions can be avoided in parallel
applications, but the ``Client`` API calls
pertaining to model actions are identical
to non-parallel applications.

.. _SR Fortran Scripts:

Scripts
=======

For an example of placing a PyTorch script in the database
and executing the script using a stored tensor,
see the :ref:`SR Parallel MPI` example.  The
aforementioned example is customized to show how
key collisions can be avoided in parallel
applications, but the ``Client`` API calls
pertaining to script actions are identical
to non-parallel applications.

.. _SR Parallel MPI:

Parallel (MPI) execution
========================

In this example, an MPI program that
sets a model, sets a script, executes a script,
executes a model, sends a tensor, and receives
a tensor is shown.  This example illustrates
how keys can be prefixed to prevent key
collisions across MPI ranks.  Note that only one
model and script are set, which is shared across
all ranks.  It is important to note that the
``Client`` API calls made in this program are
equally applicable to non-MPI programs.

This example will go step-by-step through the program and
then present the entirety of the example code at the end.

The MNIST dataset and model typically take images of
digits and quantifies how likely that number is to be 0, 1, 2,
etc.. For simplicity here, this example instead
generates random numbers to represent an image.

**Initialization**

At the top of the program, the SmartRedis Fortran client
(which is coded as a Fortran module) is imported using

.. code-block:: Fortran

  use smartredis_client, only : client_type

where ``client_type`` is a Fortran derived-type containing
the methods used to communicate with the RedisAI database.
A particular instance is declared via

.. code-block:: Fortran

  type(client_type) :: client

An initializer routine, implemented as a type-bound
procedure, must be called before any of the other
methods are used:

.. code-block:: Fortran

  return_code = client%initialize(.true.)
  if (return_code .ne. SRNoError) stop 'Error in initializing client'

The only optional argument to the initialize
routine is to determine whether the RedisAI
database is clustered (i.e. spread over a number
of nodes, ``.true.``) or exists as a single instance.

If an individual rank is expected to
send only its local data, a separate client must
be initialized on every MPI task Furthermore,
to avoid the collision of key names when running
on multiple MPI tasks, we store the rank of the
MPI process which will be used as the suffix for
all keys in this example.

On the root MPI task, two additional client methods
(``set_model_from_file`` and ``set_script_from_file``)
are called. ``set_model_from_file`` loads a saved
PyTorch model and stores it in the database using the key
``mnist_model``. Similarly, ``set_script_from_file``
loads a script that can be used to process data on the
database cluster.

.. code-block:: Fortran

  if (pe_id == 0) then
    return_code = client%set_model_from_file(model_key, model_file, "TORCH", "CPU")
    if (return_code .ne. SRNoError) stop 'Error in setting model'
    return_code = client%set_script_from_file(script_key, "CPU", script_file)
    if (return_code .ne. SRNoError) stop 'Error in setting script'
  endif

This only needs to be done on the root MPI task because
this example assumes that every rank is using the same model.
If the model is intended to be rank-specific, a unique
identifier (like the MPI rank) must be used.

At this point the initialization of the program is
complete: each rank has its own SmartRedis client,
initialized a PyTorch model has been loaded and
stored into the database with its own identifying
key, and a preprocessing script has also
been loaded and stored in the database

**Performing inference on Fortran data**

The ``run_mnist`` subroutine coordinates the inference
cycle of generating data (i.e. the synthetic MNIST image) from
the application and then the use of the client to
run a preprocessing script on data within the database and to
perform an inference from the AI model. The local
variables are declared at the top of the subroutine and are
instructive to communicate the expected shape of the
inputs to the various client methods.

.. code-block:: Fortran

  integer, parameter :: mnist_dim1 = 28
  integer, parameter :: mnist_dim2 = 28
  integer, parameter :: result_dim1 = 10

The first two integers ``mnist_dim1`` and ``mnist_dim2``
specify the shape of the input data. In the case of the
MNIST dataset, it expects a 4D tensor describing a 'picture'
of a number with dimensions [1,1,28,28] representing a
batch size (of one) and a three dimensional array. ``result_dim1``
specifies what the size of the resulting inference
will be. In this case, it is a vector of length 10, where
each element represents the probability that the data
represents a number from 0-9.

The next declaration declares the strings that will be
used to define objects representing inputs/outputs from the
scripts and inference models.

.. code-block:: Fortran

  character(len=255) :: in_key
  character(len=255) :: script_out_key
  character(len=255) :: out_key

Note that these are standard Fortran strings. However,
because the model and scripts may require the use of multiple
inputs/outputs, these will need to be converted into a
vector of strings.

.. code-block:: Fortran

    character(len=255), dimension(1) :: inputs
    character(len=255), dimension(1) :: outputs

In this case, only one input and output are expected the
vector of strings only need to be one element long. In the
case of multiple inputs/outputs, change the ``dimension``
attribute of the ``inputs`` and ``outputs`` accordingly,
e.g. for two inputs this code would be ``character(len=255),
dimension(2) :: inputs``.

Next, the input and output keys for the model and script are
now constructed

.. code-block:: Fortran

  in_key = "mnist_input_rank"//trim(key_suffix)
  script_out_key = "mnist_processed_input_rank"//trim(key_suffix)
  out_key = "mnist_processed_input_rank"//trim(key_suffix)

As mentioned previously, unique identifying keys are
constructed by including a suffix based on MPI tasks.

The subroutine, in place of an actual simulation, next
generates an array of random numbers and puts this array
into the Redis database.

.. code-block:: Fortran

  call random_number(array)
  return_code = client%put_tensor(in_key, array, shape(array))
  if (return_code .ne. SRNoError) stop 'Error putting tensor in the database'

The Redis database can now be called to run preprocessing
scripts on these data.

.. code-block:: Fortran

  inputs(1) = in_key
  outputs(1) = script_out_key
  return_code = client%run_script(script_name, "pre_process", inputs, outputs)
  if (return_code .ne. SRNoError) stop 'Error running script'

The call to ``client%run_script`` specifies the
key used to identify the script loaded during
initialization, ``pre_process`` is the name of
the function to run that is defined in that script,
and the ``inputs``/``outputs`` are the vector of
keys described previously. In this case, the call to
``run_script`` will trigger the RedisAI database
to execute ``pre_process`` on the generated data
(stored using the key ``mnist_input_rank_XX`` where ``XX``
represents the MPI rank) and storing the result of
``pre_process`` in the database as
``mnist_processed_input_rank_XX``. One key aspect to
emphasize, is that the calculations are done within the
database, not on the application side and the results
are not immediately available to the application. The retrieval
of data from the database is demonstrated next.

The data have been processed and now we can run the
inference model. The setup of the inputs/outputs
is the same as before, with the exception that the
input to the inference model, is stored using the
key ``mnist_processed_input_rank_XX``
and the output will stored using the same key.

.. code-block:: Fortran

  inputs(1) = script_out_key
  outputs(1) = out_key
  return_code = client%run_model(model_name, inputs, outputs)
  if (return_code .ne. SRNoError) stop 'Error running model'

As before the results of running the inference are
stored within the database and are not available to
the application immediately. However, we can 'retrieve'
the tensor from the database by using the ``unpack_tensor`` method.

.. code-block:: Fortran

  return_code = client%unpack_tensor(out_key, result, shape(result))
  if (return_code .ne. SRNoError) stop 'Error retrieving the tensor'

The ``result`` array now contains the outcome of the inference.
It is a 10-element array representing the likelihood that the
'image' (generated using the random numbers) is one of the numbers
[0-9].

**Key points**

The script, models, and data used here represent the
coordination of different software stacks (PyTorch, RedisAI, and
Fortran) however the application code is all written in
standard Fortran. Any operations that need to be done to
communicate with the database and exchange data are opaque
to the application.

**Source Code**

Fortran program:

.. literalinclude:: ../smartredis/examples/parallel/fortran/smartredis_mnist.F90
  :linenos:
  :language: fortran

Python Pre-Processing:

.. literalinclude:: ../smartredis/examples/common/mnist_data/data_processing_script.txt
  :linenos:
  :language: Python
  :lines: 15-20