from os import environ

"""This script checks that the names of multiple
   incoming entities is correctly passed through
   a WLM.
   SSKEYIN is the environment variable passed by
   SmartSim and it should contain two names:
   sleep_0 and sleep_1. For this test, we
   also pass them in as separate variables 
   NAME_0 and NAME_1. This program will fail if
   SSKEYIN does not contain NAME_0 and NAME_1 in.
   the right order. 
"""

sskeyin = environ["SSKEYIN"]
name_0 = environ["NAME_0"]
name_1 = environ["NAME_1"]

smartsim_separator = ","  # this is the current choice

expected_sskeyin = smartsim_separator.join((name_0, name_1))

if sskeyin != expected_sskeyin:
    raise ValueError(
        f"SSKEYIN expected to be {expected_sskeyin}, " f"but was {sskeyin}"
    )
