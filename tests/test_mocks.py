from unittest.mock import MagicMock

import pytest
import sys
import uuid


class ProcessGroupMock(MagicMock):
    puids = [121, 122]


def test_mocked_backend(monkeypatch: pytest.MonkeyPatch):
    system_mock = MagicMock(nodes=["node1", "node2", "node3"])
    process_group_mock = ProcessGroupMock()

    # node_mock = lambda: MagicMock(num_cpus=4, num_gpus=2, ident=str(uuid.uuid4()))

    monkeypatch.setitem(
        sys.modules,
        "dragon",
        MagicMock(**{"data.ddict.ddict": MagicMock(**{"DDict": MagicMock()})}),
    )

    # monkeypatch.setitem(
    #     sys.modules, "dragon.data.ddict.ddict", MagicMock(**{"DDict": MagicMock()})
    # )
    monkeypatch.setitem(sys.modules, "dragon.infrastructure.connection", MagicMock())
    monkeypatch.setitem(
        sys.modules,
        "dragon.infrastructure.policy",
        MagicMock(**{"Policy.return_value": MagicMock()}),
    )
    monkeypatch.setitem(sys.modules, "dragon.infrastructure.process_desc", MagicMock())
    monkeypatch.setitem(sys.modules, "dragon.native.group_state", MagicMock())
    monkeypatch.setitem(
        sys.modules,
        "dragon.native.machine",
        MagicMock(
            **{
                "System.return_value": system_mock,
                "Node.return_value": MagicMock(
                    num_cpus=4, num_gpus=2, ident=str(uuid.uuid4())
                ),
            }
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "dragon.native.process",
        MagicMock(**{"Process": MagicMock(returncode=0)}),
    )
    monkeypatch.setitem(
        sys.modules,
        "dragon.native.process_group",
        MagicMock(**{"Process.return_value": process_group_mock}),
    )

    import dragon.data.ddict.ddict as dragon_dict
    from smartsim._core.launcher.dragon.dragonBackend import DragonBackend

    dd = dragon_dict.DDict()
    dd["foo"] = "bar"

    backend = DragonBackend(42)
    assert backend._pid == 42
