from os import getcwd, name

import pytest

from smartsim import Experiment
from smartsim.entity import EntityList
from smartsim.settings import RunSettings


def test_entity_list_init():
    with pytest.raises(NotImplementedError):
        ent_list = EntityList("list", getcwd(), perm_strat="all_perm")


def test_entity_list_getitem():
    """EntityList.__getitem__ is overridden in Ensemble, so we had to pass an instance of Ensemble
    to EntityList.__getitem__ in order to add test coverage to EntityList.__getitem__
    """
    exp = Experiment("name")
    ens_settings = RunSettings("python")
    ensemble = exp.create_ensemble("name", replicas=4, run_settings=ens_settings)
    assert ensemble.__getitem__("name_3") == ensemble["name_3"]
