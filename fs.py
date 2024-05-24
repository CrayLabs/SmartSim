import logging
import multiprocessing as mp
import time

import dragon
import dragon.channels as dch
import dragon.infrastructure.facts as df
import dragon.infrastructure.parameters as dp
import dragon.managed_memory as dm
import dragon.utils as du
from dragon.data.distdictionary.dragon_dict import DragonDict

import os

import dragon.infrastructure

pid = os.getpid()
logger = logging.getLogger()
logging.basicConfig(level="DEBUG", filename=f"log.{pid}.txt", filemode="w+")


def read_key_from_another_process(
    feature_store: DragonDict,
    key: str,
) -> None:
    """
    Create a static set of worker resources required to support MLI features
    """
    for i in range(10):
        try:
            logger.info(f"read_key_from_another_process::{key}")
            value = feature_store[key]
            logger.info(f"retrieved value using key `{key}`: {value}")
        except:
            logger.exception("read_key_from_another_process failed")
        finally:
            time.sleep(2)


def main():
    mp.set_start_method("dragon")
    logger.info("dragon mp configured")

    feature_store = DragonDict(1, 5, 1024 * 1024 * 1024)

    logger.debug("Storing value in feature store")
    key = "my_key"
    feature_store[key] = "foo"

    try:
        logger.info(f"Attempting to retrieve feature store key: `key`")
        value = feature_store[key]
        logger.info(f"Feature store key retrieved: `{key}`. Value: {value}")
    except:
        logger.error("Retrieving key failed.", exc_info=True)

    logger.debug("Attaching to memory pool")
    reader_process = mp.Process(
        target=read_key_from_another_process, args=(feature_store, key)
    )
    reader_process.start()


if __name__ == "__main__":
    main()
