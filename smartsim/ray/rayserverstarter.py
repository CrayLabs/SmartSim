from smartsim.ray import RayServer
import argparse


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

print("OK, YOU ARE ON THE RIGHT TRACK BUDDY. I LOVE YOU.")
server = RayServer(current_ip(), args.zmq_port, args.ray_port, args.ray_password, args.ray_num_cpus)
server.start()
