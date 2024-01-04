import os
import sys

var_name = sys.argv[1]
env_value = os.environ.get(sys.argv[1], None)

if env_value:
    print(f"{var_name}=={env_value}")
    sys.exit(0)

print("env var not found")
