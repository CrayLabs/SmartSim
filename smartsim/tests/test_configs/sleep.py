import time
import argparse

def sleep(time_to_sleep):
    print("Starting sleep.py", flush=True)
    time.sleep(time_to_sleep)
    print(str(time_to_sleep), flush=True)
    print("done", flush=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=int, default=15)
    args = parser.parse_args()
    sleep(args.time)



