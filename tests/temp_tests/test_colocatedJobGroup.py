from smartsim.entity.model import Application
from smartsim.launchable.basejob import BaseJob
from smartsim.launchable.colocatedJobGroup import ColocatedJobGroup
from smartsim.launchable.job import Job


# TODO replace with LaunchSettings
class RunSettings:
    pass


app_1 = Application("app_1", "python", RunSettings())
app_2 = Application("app_2", "python", RunSettings())
app_3 = Application("app_3", "python", RunSettings())


def test_create_ColocatedJobGroup():
    job_1 = BaseJob()
    job_group = ColocatedJobGroup([job_1])
    assert len(job_group) == 1


def test_getitem_ColocatedJobGroup():
    job_1 = Job(app_1, RunSettings())
    job_2 = Job(app_2, RunSettings())
    job_group = ColocatedJobGroup([job_1, job_2])
    get_value = job_group[0].entity.name
    assert get_value == job_1.entity.name


def test_setitem_JobGroup():
    job_1 = Job(app_1, RunSettings())
    job_2 = Job(app_2, RunSettings())
    job_group = ColocatedJobGroup([job_1, job_2])
    job_3 = Job(app_3, RunSettings())
    job_group[1] = job_3
    assert len(job_group) == 2
    get_value = job_group[1].entity.name
    assert get_value == job_3.entity.name


def test_delitem_ColocatedJobGroup():
    job_1 = BaseJob()
    job_2 = BaseJob()
    job_group = ColocatedJobGroup([job_1, job_2])
    assert len(job_group) == 2
    del job_group[1]
    assert len(job_group) == 1


def test_len_ColocatedJobGroup():
    job_1 = BaseJob()
    job_2 = BaseJob()
    job_group = ColocatedJobGroup([job_1, job_2])
    assert len(job_group) == 2


def test_insert_ColocatedJobGroup():
    job_1 = BaseJob()
    job_2 = BaseJob()
    job_group = ColocatedJobGroup([job_1, job_2])
    job_3 = BaseJob()
    job_group.insert(0, job_3)
    get_value = job_group[0]
    assert get_value == job_3
