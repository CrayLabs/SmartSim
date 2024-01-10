import sys
import time


def main() -> int:
    print(";START;")
    time.sleep(20)
    print(";MID;")
    print("This is an error msg", file=sys.stderr)
    time.sleep(20)
    print(";END;")

    print("yay!!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
