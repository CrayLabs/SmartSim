
import argparse
from smartsim.remote.cmdServer import CMDServer

parser = argparse.ArgumentParser()
parser.add_argument('--address',
                    type=str,
                    default="127.0.0.1",
                    help='TCP address of the Command Server to be launched')
parser.add_argument('--port',
                    type=int,
                    default=5555,
                    help='Port of the Command Server to be launched')
args = parser.parse_args()

# start the Command Server
cmd_center = CMDServer(args.address, args.port)
cmd_center.serve()