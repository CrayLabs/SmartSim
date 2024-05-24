from gc import callbacks
import argparse
import logging
import multiprocessing as mp
import pathlib
import random
import subprocess
import sys
import shutil
import time
import typing as t
import uuid

import dragon
import dragon.channels as dch
import dragon.infrastructure.facts as df
import dragon.infrastructure.parameters as dp
import dragon.managed_memory as dm
import dragon.utils as du
from dragon.data.distdictionary.dragon_dict import DragonDict

import dragon.infrastructure
import base64

import infrastructure as mli


class ReaderArgs:
    def __init__(self, pool_id: str, channel: int, output_dir: str) -> None:
        self.pool_id: bytes = du.B64.str_to_bytes(pool_id)
        self.channel: int = channel
        self.output_dir: pathlib.Path = pathlib.Path(output_dir)


def get_args() -> ReaderArgs:
    parser = argparse.ArgumentParser()

    parser.add_argument("poolid")
    parser.add_argument("channel", type=int, default=df.BASE_USER_MANAGED_CUID)

    args = parser.parse_args(sys.argv[1:], type=str)

    return ReaderArgs(args.pool_id, args.channel)


def main(args: ReaderArgs):
    mli.configure_multiprocessing()

    mem_pool = dm.MemoryPool.attach(args.pool_id)

    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True, exist_ok=True)

    callback_channel = dch.Channel(mem_pool, args.channel_id)
    reader = dch.ChannelRecvH(callback_channel)
    reader.open()
    running = True

    while running:
        try:
            content = reader.recv_bytes(timeout=5, blocking=True)
            ts = time.time_ns()

            with open(f"mli/reader_{ts}.out", "w+") as fp:
                fp.write(f"{content=}\n")
        except:
            running = False


if __name__ == "__main__":
    args = get_args()
    main(args)
