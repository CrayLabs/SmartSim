from smartsim.entity.model import Application
from smartsim.launchable.basejob import BaseJob
from smartsim.launchable.job import Job
from smartsim.launchable.jobGroup import JobGroup
from smartsim.settings.launchSettings import LaunchSettings

# TODO replace with LaunchSettings
app_1 = Application("app_1", "python", LaunchSettings("slurm"))
app_2 = Application("app_2", "python", LaunchSettings("slurm"))
app_3 = Application("app_3", "python", LaunchSettings("slurm"))


def test_create_JobGroup():
    job_1 = BaseJob()
    job_group = JobGroup([job_1])
    assert len(job_group) == 1


def test_getitem_JobGroup():
    job_1 = Job(app_1, LaunchSettings("slurm"))
    job_2 = Job(app_2, LaunchSettings("slurm"))
    job_group = JobGroup([job_1, job_2])
    get_value = job_group[0].entity.name
    assert get_value == job_1.entity.name


def test_setitem_JobGroup():
    job_1 = Job(app_1, LaunchSettings("slurm"))
    job_2 = Job(app_2, LaunchSettings("slurm"))
    job_group = JobGroup([job_1, job_2])
    job_3 = Job(app_3, LaunchSettings("slurm"))
    job_group[1] = job_3
    assert len(job_group) == 2
    get_value = job_group[1]
    assert get_value.entity.name == job_3.entity.name


def test_delitem_JobGroup():
    job_1 = BaseJob()
    job_2 = BaseJob()
    job_group = JobGroup([job_1, job_2])
    assert len(job_group) == 2
    del job_group[1]
    assert len(job_group) == 1


def test_len_JobGroup():
    job_1 = BaseJob()
    job_2 = BaseJob()
    job_group = JobGroup([job_1, job_2])
    assert len(job_group) == 2


def test_insert_JobGroup():
    job_1 = BaseJob()
    job_2 = BaseJob()
    job_group = JobGroup([job_1, job_2])
    job_3 = BaseJob()
    job_group.insert(0, job_3)
    get_value = job_group[0]
    assert get_value == job_3
