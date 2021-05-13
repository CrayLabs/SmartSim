from smartsim.ray import RayServer
import setproctitle
import argparse
import os

parser = argparse.ArgumentParser(description="SmartSim Ray server launcher")
parser.add_argument("--ray-port", type=int, help="Port used by Ray to start the Redis server at")
parser.add_argument("--ray-password", type=str, help="Password of Redis cluster")
parser.add_argument("--zmq-port", type=int, help="Port used by ZMQ")
parser.add_argument("--ray-num-cpus", type=int, help="Number of CPUs to be used by Ray head node worker")
args = parser.parse_args()

def current_ip():
    import socket
    hostname = socket.getfqdn(socket.gethostname())
    return socket.gethostbyname(hostname)

os.environ["TUNE_MAX_PENDING_TRIALS_PG"]="32"
# os.environ["TUNE_STATE_REFRESH_PERIOD"]="1"
# os.environ["TUNE_PLACEMENT_GROUP_AUTO_DISABLED"]="1"
# os.environ["TUNE_FUNCTION_THREAD_TIMEOUT_S"]="5"
# os.environ["TUNE_RESULT_BUFFER_LENGTH"]="0"
# os.environ["TUNE_TRIAL_STARTUP_GRACE_PERIOD"]="10"
# os.environ["TUNE_RESULT_BUFFER_MAX_TIME_S"]="0"
# os.environ["TUNE_PLACEMENT_GROUP_WAIT_S"]="5"
# os.environ["TUNE_PLACEMENT_GROUP_RECON_INTERVAL"]="1"
# os.environ["RAY_BACKEND_LOG_LEVEL"]="0"

#os.environ["OMP_NUM_THREADS"] = "72"

server = RayServer(current_ip(), args.zmq_port, args.ray_port, args.ray_password, args.ray_num_cpus)
setproctitle.setproctitle("RayServer")
server.start()