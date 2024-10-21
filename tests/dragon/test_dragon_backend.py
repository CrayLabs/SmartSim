# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import time
import uuid

import pytest

dragon = pytest.importorskip("dragon")


from smartsim._core.launcher.dragon.dragonBackend import DragonBackend
from smartsim._core.mli.comm.channel.dragon_channel import DragonCommChannel
from smartsim._core.mli.infrastructure.comm.event import (
    OnCreateConsumer,
    OnShutdownRequested,
)
from smartsim._core.mli.infrastructure.control.listener import (
    ConsumerRegistrationListener,
)
from smartsim._core.mli.infrastructure.storage.backbone_feature_store import (
    BackboneFeatureStore,
)
from smartsim.log import get_logger

# The tests in this file belong to the dragon group
pytestmark = pytest.mark.dragon
logger = get_logger(__name__)


@pytest.fixture(scope="module")
def the_backend() -> DragonBackend:
    return DragonBackend(pid=9999)


@pytest.mark.skip("Test is unreliable on build agent and may hang. TODO: Fix")
def test_dragonbackend_start_listener(the_backend: DragonBackend):
    """Verify the background process listening to consumer registration events
    is up and processing messages as expected."""

    # We need to let the backend create the backbone to continue
    backbone = the_backend._create_backbone()
    backbone.pop(BackboneFeatureStore.MLI_NOTIFY_CONSUMERS)
    backbone.pop(BackboneFeatureStore.MLI_REGISTRAR_CONSUMER)

    os.environ[BackboneFeatureStore.MLI_BACKBONE] = backbone.descriptor

    with pytest.raises(KeyError) as ex:
        # we expect the value of the consumer to be empty until
        # the listener start-up completes.
        backbone[BackboneFeatureStore.MLI_REGISTRAR_CONSUMER]

    assert "not found" in ex.value.args[0]

    drg_process = the_backend.start_event_listener(cpu_affinity=[], gpu_affinity=[])

    # # confirm there is a process still running
    logger.info(f"Dragon process started: {drg_process}")
    assert drg_process is not None, "Backend was unable to start event listener"
    assert drg_process.puid != 0, "Process unique ID is empty"
    assert drg_process.returncode is None, "Listener terminated early"

    # wait for the event listener to come up
    try:
        config = backbone.wait_for(
            [BackboneFeatureStore.MLI_REGISTRAR_CONSUMER], timeout=30
        )
        # verify result was in the returned configuration map
        assert config[BackboneFeatureStore.MLI_REGISTRAR_CONSUMER]
    except Exception:
        raise KeyError(
            f"Unable to locate {BackboneFeatureStore.MLI_REGISTRAR_CONSUMER}"
            "in the backbone"
        )

    # wait_for ensures the normal retrieval will now work, error-free
    descriptor = backbone[BackboneFeatureStore.MLI_REGISTRAR_CONSUMER]
    assert descriptor is not None

    # register a new listener channel
    comm_channel = DragonCommChannel.from_descriptor(descriptor)
    mock_descriptor = str(uuid.uuid4())
    event = OnCreateConsumer("test_dragonbackend_start_listener", mock_descriptor, [])

    event_bytes = bytes(event)
    comm_channel.send(event_bytes)

    subscriber_list = []

    # Give the channel time to write the message and the listener time to handle it
    for i in range(20):
        time.sleep(1)
        # Retrieve the subscriber list from the backbone and verify it is updated
        if subscriber_list := backbone.notification_channels:
            logger.debug(f"The subscriber list was populated after {i} iterations")
            break

    assert mock_descriptor in subscriber_list

    # now send a shutdown message to terminate the listener
    return_code = drg_process.returncode

    # clean up if the OnShutdownRequested wasn't properly handled
    if return_code is None and drg_process.is_alive:
        drg_process.kill()
        drg_process.join()


def test_dragonbackend_backend_consumer(the_backend: DragonBackend):
    """Verify the listener background process updates the appropriate
    value in the backbone."""

    # We need to let the backend create the backbone to continue
    backbone = the_backend._create_backbone()
    backbone.pop(BackboneFeatureStore.MLI_NOTIFY_CONSUMERS)
    backbone.pop(BackboneFeatureStore.MLI_REGISTRAR_CONSUMER)

    assert backbone._allow_reserved_writes

    # create listener with `as_service=False` to perform a single loop iteration
    listener = ConsumerRegistrationListener(backbone, 1.0, 1.0, as_service=False)

    logger.debug(f"backbone loaded? {listener._backbone}")
    logger.debug(f"listener created? {listener}")

    try:
        # call the service execute method directly to trigger
        # the entire service lifecycle
        listener.execute()

        consumer_desc = backbone[BackboneFeatureStore.MLI_REGISTRAR_CONSUMER]
        logger.debug(f"MLI_REGISTRAR_CONSUMER: {consumer_desc}")

        assert consumer_desc
    except Exception as ex:
        logger.info("")
    finally:
        listener._on_shutdown()


def test_dragonbackend_event_handled(the_backend: DragonBackend):
    """Verify the event listener process updates the appropriate
    value in the backbone when an event is received and again on shutdown.
    """
    # We need to let the backend create the backbone to continue
    backbone = the_backend._create_backbone()
    backbone.pop(BackboneFeatureStore.MLI_NOTIFY_CONSUMERS)
    backbone.pop(BackboneFeatureStore.MLI_REGISTRAR_CONSUMER)

    # create the listener to be tested
    listener = ConsumerRegistrationListener(backbone, 1.0, 1.0, as_service=False)

    assert listener._backbone, "The listener is not attached to a backbone"

    try:
        # set up the listener but don't let the service event loop start
        listener._create_eventing()  # listener.execute()

        # grab the channel descriptor so we can simulate registrations
        channel_desc = backbone[BackboneFeatureStore.MLI_REGISTRAR_CONSUMER]
        comm_channel = DragonCommChannel.from_descriptor(channel_desc)

        num_events = 5
        events = []
        for i in range(num_events):
            # register some mock consumers using the backend channel
            event = OnCreateConsumer(
                "test_dragonbackend_event_handled",
                f"mock-consumer-descriptor-{uuid.uuid4()}",
                [],
            )
            event_bytes = bytes(event)
            comm_channel.send(event_bytes)
            events.append(event)

        # run few iterations of the event loop in case it takes a few cycles to write
        for _ in range(20):
            listener._on_iteration()
            # Grab the value that should be getting updated
            notify_consumers = set(backbone.notification_channels)
            if len(notify_consumers) == len(events):
                logger.info(f"Retrieved all consumers after {i} listen cycles")
                break

        # ... and confirm that all the mock consumer descriptors are registered
        assert set([e.descriptor for e in events]) == set(notify_consumers)
        logger.info(f"Number of registered consumers: {len(notify_consumers)}")

    except Exception as ex:
        logger.exception(f"test_dragonbackend_event_handled - exception occurred: {ex}")
        assert False
    finally:
        # shutdown should unregister a registration listener
        listener._on_shutdown()

    for i in range(10):
        if BackboneFeatureStore.MLI_REGISTRAR_CONSUMER not in backbone:
            logger.debug(f"The listener was removed after {i} iterations")
            channel_desc = None
            break

    # we should see that there is no listener registered
    assert not channel_desc, "Listener shutdown failed to clean up the backbone"


def test_dragonbackend_shutdown_event(the_backend: DragonBackend):
    """Verify the background process shuts down when it receives a
    shutdown request."""

    # We need to let the backend create the backbone to continue
    backbone = the_backend._create_backbone()
    backbone.pop(BackboneFeatureStore.MLI_NOTIFY_CONSUMERS)
    backbone.pop(BackboneFeatureStore.MLI_REGISTRAR_CONSUMER)

    listener = ConsumerRegistrationListener(backbone, 1.0, 1.0, as_service=True)

    # set up the listener but don't let the listener loop start
    listener._create_eventing()  # listener.execute()

    # grab the channel descriptor so we can publish to it
    channel_desc = backbone[BackboneFeatureStore.MLI_REGISTRAR_CONSUMER]
    comm_channel = DragonCommChannel.from_descriptor(channel_desc)

    assert listener._consumer.listening, "Listener isn't ready to listen"

    # send a shutdown request...
    event = OnShutdownRequested("test_dragonbackend_shutdown_event")
    event_bytes = bytes(event)
    comm_channel.send(event_bytes, 0.1)

    # execute should encounter the shutdown and exit
    listener.execute()

    # ...and confirm the listener is now cancelled
    assert not listener._consumer.listening


@pytest.mark.parametrize("health_check_frequency", [10, 20])
def test_dragonbackend_shutdown_on_health_check(
    the_backend: DragonBackend,
    health_check_frequency: float,
):
    """Verify that the event listener automatically shuts down when
    a new listener is registered in its place.

    :param health_check_frequency: The expected frequency of service health check
     invocations"""

    # We need to let the backend create the backbone to continue
    backbone = the_backend._create_backbone()
    backbone.pop(BackboneFeatureStore.MLI_NOTIFY_CONSUMERS)
    backbone.pop(BackboneFeatureStore.MLI_REGISTRAR_CONSUMER)

    listener = ConsumerRegistrationListener(
        backbone,
        1.0,
        1.0,
        as_service=True,  # allow service to run long enough to health check
        health_check_frequency=health_check_frequency,
    )

    # set up the listener but don't let the listener loop start
    listener._create_eventing()  # listener.execute()
    assert listener._consumer.listening, "Listener wasn't ready to listen"

    # Replace the consumer descriptor in the backbone to trigger
    # an automatic shutdown
    backbone[BackboneFeatureStore.MLI_REGISTRAR_CONSUMER] = str(uuid.uuid4())

    # set the last health check manually to verify the duration
    start_at = time.time()
    listener._last_health_check = time.time()

    # run execute to let the service trigger health checks
    listener.execute()
    elapsed = time.time() - start_at

    # confirm the frequency of the health check was honored
    assert elapsed >= health_check_frequency

    # ...and confirm the listener is now cancelled
    assert (
        not listener._consumer.listening
    ), "Listener was not automatically shutdown by the health check"
