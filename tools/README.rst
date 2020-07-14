***************
Code generators
***************

Why Generate Code?
==================

The C++ SmartSim client handles the low-level interactions between Redis and
protobuf. The high-level, user-facing functions are implemented as methods of
the SmartSimClient class. The C client routines (which the Fortran client
relies on) are primarily wrappers to those methods. Due to limitations of the C
and Fortran languages, separate routines must be created for every type of
variable (e.g. two routines are needed for 32-bit and 64-bit integers).

One of the anticipated cases where are user might need to add support for a
new variable type. In addition to adding a new method to the C++ client, a
user would also have to have make C and Fortran wrappers. The C and Fortran
code generators here help automate this task and ensure consistency of functions
for all variable types.

Adding a New Variable Type
==========================

This example shows how the generator scripts would have to be modified to add
support in SmartSim for a :code:`boolean` variable.

1. Make a new protobuf template for a C++ :code:`bool` and make class methods
   for each of the main SmartSim functionals
2. Modify :code:`c_code_generator.py`, adding :code:`bool` to the
   :code:`var_types` list.
3. Modify :code:`fortran_code_generator.py` adding :code:`{'c_kind':'c_bool' 'ftype':'logical', 'kind':4}` to the :code:`var_types` list.
4. Run both scripts and recompile the SmartSim clients.

Adding a New Function
=====================

.. note::
    If the new routine does not follow the existing routines, more extensive
    modifications may have to be made to the code generator.

1. Add the new method to the C++ client
2. Add a new instance of :code:`c_function` to :code:`c_code_generator.py`
3. Modify :code:`fortran_code_generator.py`
    a. Create a new :code:`fortran_procedure` that should be the generic name of
       the function/routine
    b. In the loop over :code:`var_types`, append a new :code:`fortran_routine`
       to the procedure.
4. Run both scripts and recompile the SmartSim clients