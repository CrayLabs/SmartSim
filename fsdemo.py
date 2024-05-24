from urllib import request
import dragon
import argparse
import multiprocessing as mp
from dragon.data.distdictionary.dragon_dict import DragonDict
import requests

from smartsim._core.entrypoints import indirect


def _retrieve_value(_dict, key, client_id):
    value = _dict[key]
    print(
        f"Retrieving value:{value} for key:{key} for client id:{client_id} from the dictionary",
        flush=True,
    )
    return value


def _store_key_value(_dict, key, value, client_id):
    print(
        f"Storing key:{key} and value:{value} from client id:{client_id} into the dictionary",
        flush=True,
    )
    _dict[key] = value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed dictionary example")
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=2,
        help="number of nodes the dictionary distributed across",
    )
    parser.add_argument(
        "--managers_per_node",
        type=int,
        default=1,
        help="number of managers per node for the dragon dict",
    )
    parser.add_argument(
        "--total_mem_size",
        type=int,
        default=1,
        help="total managed memory size for dictionary in GB",
    )

    my_args = parser.parse_args()
    mp.set_start_method("dragon")

    # Instantiate the dictionary and start the processes
    total_mem_size = my_args.total_mem_size * (1024 * 1024 * 1024)
    dd = DragonDict(my_args.managers_per_node, my_args.num_nodes, total_mem_size)

    client_proc_1 = mp.Process(target=_store_key_value, args=(dd, "Hello", "Dragon", 1))
    client_proc_1.start()
    client_proc_1.join()

    for i in range(my_args.num_nodes - 1):
        id = i + 2
        client_proc_2 = mp.Process(target=_retrieve_value, args=(dd, "Hello", id))
        client_proc_2.start()
        client_proc_2.join()

    # client_proc_1 = mp.Process(target=_store_key_value, args=(dd, "Hello", "Dragon", 1))
    # client_proc_1.start()
    # client_proc_1.join()

    # client_proc_2 = mp.Process(target=_retrieve_value, args=(dd, "Hello", 2))
    # client_proc_2.start()
    # client_proc_2.join()

    print("Done here. Closing the Dragon Dictionary", flush=True)
    dd.close()




first inference request
1. actual ml model
2. input tensor
3. tensor dimensions
4. order


    - RESPONSE- 
    will include a _KEY_ for you to send on all subsequent requests


second inference request
1. NOPE -> instead ml model ref key....
2. same
3. NOPE -> shouldn't need it
2. NOPE -> shouldn't need it


Registration
1. here's a model
2. here's the input shape

    - reg response -
    1. here's a key

ISSUE - how do i differentiate between a semi-direct inference AND an indirect


do we really want to have clients pull data out of messages in channels?
- why shouldn't THEY just grab from dragon dict like the handler would have to?



registration of model ->
    create a worker with that definition and tell "control" about itself
        -> "here i am, talk to me using this queue"

1. ask for model by name/key
    2. get back a "ModelRegistration" that identifies worker/queue

3. (cache and) Use registration info to send request 


SO -> I really think we should separate registration and inference
    - to avoid weird schemas and resending the same data over & over

