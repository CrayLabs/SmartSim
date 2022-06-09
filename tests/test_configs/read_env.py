import os

if os.getenv('SMARTSIM_TEST_ENV'):
    print('SMARTSIM_TEST_ENV was set')
else:
    raise print('SMARTSIM_TEST_ENV was not set')


