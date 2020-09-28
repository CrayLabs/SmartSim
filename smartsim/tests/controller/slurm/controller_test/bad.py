
import time
import argparse


def divide_by_zero(time_to_wait):
    """A sample program to test error handling at different points in execution"""
    time.sleep(time_to_wait)
    print(1/0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=int, default=0)
    args = parser.parse_args()
    divide_by_zero(args.time)



