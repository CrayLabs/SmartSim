# isort: off
import dragon
from dragon import fli
from dragon.channels import Channel
import dragon.channels
from dragon.data.ddict.ddict import DDict
from dragon.globalservices.api_setup import connect_to_infrastructure
from dragon.utils import b64decode

# isort: on

import argparse
import io
import numpy
import os
import time
import torch
from smartsim._core.mli.message_handler import MessageHandler


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Mock application")
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    connect_to_infrastructure()
    ddict_str = os.environ["SS_DRG_DDICT"]

    ddict = DDict.attach(ddict_str)

    to_worker_fli_str = None

    while to_worker_fli_str is None:
        try:
            to_worker_fli_str = ddict["to_worker_fli"]
        except Exception as e:
            time.sleep(1)

    to_worker_fli = fli.FLInterface.attach(b64decode(to_worker_fli_str))

    batch_size = 32
    model = torch.jit.load(f"resnet50.{args.device.upper()}.pt")
    buffer = io.BytesIO()
    batch = torch.randn((batch_size, 3, 224, 224), dtype=torch.float32)
    scripted = torch.jit.trace(model, batch)
    torch.jit.save(scripted, buffer)

    total_iterations = 10

    headers=[
                "batch_size",
                "build_tensor",
                "build_request",
                "serialize_request",
                "send",
                "receive",
                "deserialize_response",
                "deserialize_tensor",
            ]

    print(",".join(headers))

    for batch_size in [1, 8, 32, 64, 128]:

        timings = []
        for iteration_number in range(total_iterations + int(batch_size==1)):

            timings.append([batch_size])

            batch = torch.randn((batch_size, 3, 224, 224), dtype=torch.float32)

            start = time.perf_counter()
            interm = start
            built_tensor = MessageHandler.build_tensor(
                batch.numpy(), "c", "float32", list(batch.shape)
            )
            timings[-1].append(time.perf_counter() - interm)
            interm = time.perf_counter()

            from_worker_ch = Channel.make_process_local()

            request = MessageHandler.build_request(
                reply_channel=from_worker_ch.serialize(),
                model=buffer.getvalue(),
                inputs=[built_tensor],
                outputs=[],
                output_descriptors=[],
                custom_attributes=None,
            )

            timings[-1].append(time.perf_counter() - interm)
            interm = time.perf_counter()
            request_bytes = MessageHandler.serialize_request(request)
            timings[-1].append(time.perf_counter() - interm)
            interm = time.perf_counter()
            with to_worker_fli.sendh(timeout=None) as to_sendh:
                to_sendh.send_bytes(request_bytes)
                timings[-1].append(time.perf_counter() - interm)
                interm = time.perf_counter()

            with from_worker_ch.recvh(timeout=None) as from_recvh:
                resp = from_recvh.recv_bytes(timeout=None)
                timings[-1].append(time.perf_counter() - interm)
                interm = time.perf_counter()
                response = MessageHandler.deserialize_response(resp)
                timings[-1].append(time.perf_counter() - interm)
                interm = time.perf_counter()
                result = torch.from_numpy(
                    numpy.frombuffer(
                        response.result.data[0].blob,
                        dtype=str(response.result.data[0].tensorDescriptor.dataType),
                    )
                )

                timings[-1].append(time.perf_counter() - interm)
                interm = time.perf_counter()

            # duration = time.perf_counter() - start
            # print(f"{duration:.3f} s")

            print(",".join(str(timing) for timing in timings[-1]))
